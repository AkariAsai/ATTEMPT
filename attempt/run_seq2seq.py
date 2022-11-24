# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
from utils import modify_model_after_init, save_training_config, save_prompts
import shutil
import glob
from data import AutoPostProcessor
from third_party.models import T5Config, T5ForConditionalGeneration
from dataclasses import dataclass, field
from options import AdapterTrainingArguments, ModelArguments, DataTrainingArguments, TrainingArguments
from third_party.trainers import Seq2SeqTrainer
from data import TaskDataCollatorForSeq2Seq
from data import AutoTask
from utils import get_adapter_config
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup
)
import transformers
from datasets import concatenate_datasets
from typing import Optional, List
import subprocess
import sys
import functools
import logging
import numpy as np
from pytz import common_timezones
import torch
import os
from torch import nn

from data.tasks import TASK_MAPPING
from metrics.metrics import TASK_TO_METRICS
from metrics.metrics import build_compute_metrics_fn

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


logger = logging.getLogger(__name__)


def run_command(command):
    output = subprocess.getoutput(command)
    return output


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               AdapterTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            pass
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load a model config
    config = T5Config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.train_task_adapters = adapter_args.train_task_adapters
    config.prefix_tuning = adapter_args.prefix_tuning
    config.attn_prefix_tuning = model_args.attn_prefix_tuning
    config.attn_method = model_args.attn_method
    config.ignore_target = model_args.ignore_target
    config.shared_attn = model_args.shared_attn
    config.prefix_num = model_args.prefix_num
    config.num_target = len(data_args.task_name)
    config.temperature = model_args.temperature
    config.learned_temperature = model_args.learned_temperature
    config.fix_attention = model_args.fix_attention
    adapter_config = get_adapter_config(
        adapter_args, data_args, training_args, config)

    # Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Initialize the model
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        adapter_config=adapter_config
    )

    if model_args.load_prefix_embeddings is True:
        if model_args.prompt_embedding_path is None:
            for name, param in model.named_parameters():
                if "prefix_shared" in name or "prefix" in name:
                    shared_params = [param]
        else:
            shared_params = []
            for path in model_args.prompt_embedding_path:
                shared_param = torch.load(path)
                shared_params.append(shared_param)
            if model_args.target_prompt_embedding_path is not None:
                target_prompt_embedding = torch.load(
                    model_args.target_prompt_embedding_path)

        if model_args.attn_prefix_tuning is True:
            if training_args.do_train is True and model_args.multi_task is False and model_args.shared_attn is False:
                # Initialize the prompt embeddings using the first prompts
                # Load all of the target prompts
                model.store_prefix_weights(shared_params)
                model.update_prefix_weights_single(shared_params[0])
            elif training_args.do_train is True and model_args.multi_task is False and model_args.shared_attn is True:
                # initialize the embeddings
                # initialize multiple shared embeddings
                model.store_prefix_weights(shared_params)
                model.update_prefix_weights_multi(
                    shared_params[0], num_target=config.num_target)
            else:
                # Load prompt embeddings except for the last one
                # Load last prompt embeddings to initialize the target prompt embeddings.
                model.store_prefix_weights(shared_params)
                model.update_prefix_weights_single(shared_params[-1])

        else:
            if model_args.target_prompt_embedding_path is None:
                model.update_prefix_weights(shared_params)
            else:
                model.update_prefix_weights(
                    shared_params, target_prompt_embedding)

    if model_args.load_attention is True and model_args.attn_path is not None:
        model.update_attention_weights(torch.load(model_args.attn_path))

    if model_args.load_attention is True and model_args.attn_path_sub is not None:
        model.update_attention_weights_sub(model_args.attn_path_sub)

    if model_args.load_layer_norm is True and model_args.layer_norm_dir is not None:
        model.update_layer_norm_weights(model_args.layer_norm_dir)

    model.resize_token_embeddings(len(tokenizer))
    model = modify_model_after_init(
        model, training_args, adapter_args, adapter_config)

    data_args.dataset_name = data_args.task_name
    data_args.eval_dataset_name = data_args.eval_dataset_name
    data_args.test_dataset_name = data_args.test_dataset_name
    data_args.dataset_config_name = data_args.dataset_config_name
    data_args.eval_dataset_config_name = data_args.eval_dataset_config_name
    data_args.test_dataset_config_name = data_args.test_dataset_config_name
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(
            data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(
            data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples, max_target_length, task_id=None):
        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["extra_fields"] = examples['extra_fields']
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['extra_fields']]

        return model_inputs

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}
    if training_args.do_train:
        # Load datasets from files if your target datasets are not in huggingface datasets.
        if data_args.train_files is not None:
            train_datasets = [AutoTask.get(dataset_name,
                                           dataset_config_name,
                                           seed=data_args.data_seed).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_train_samples, lang=data_args.lang_name, file_name=train_file)
                for dataset_name, dataset_config_name, train_file
                in zip(data_args.dataset_name, data_args.dataset_config_name, data_args.train_files)]
        else:
            train_datasets = [AutoTask.get(dataset_name,
                                           dataset_config_name,
                                           seed=data_args.data_seed).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_train_samples, lang=data_args.lang_name, file_name=data_args.train_file)
                for dataset_name, dataset_config_name
                in zip(data_args.dataset_name, data_args.dataset_config_name)]

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length, )
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]

        for i, train_dataset in enumerate(train_datasets):
            if model_args.shared_attn is True:
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(
                        preprocess_function, max_target_length=max_target_lengths[i], task_id=i),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if train_dataset != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(preprocess_function,
                                      max_target_length=max_target_lengths[i]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if train_dataset != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
        train_dataset = concatenate_datasets(train_datasets)

    if training_args.do_eval:
        if data_args.validation_files is not None:
            eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
                                                        seed=data_args.data_seed).get(
                split="validation",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=validation_file)
                for eval_dataset, eval_dataset_config, validation_file in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name, data_args.validation_files)}
        else:
            eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
                                                        seed=data_args.data_seed).get(
                split="validation",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=data_args.validation_file)
                for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]

        for k, name in enumerate(eval_datasets):
            if model_args.shared_attn is True:
                eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(
                        preprocess_function, max_target_length=max_target_lengths[k], task_id=k),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if name != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function,
                                      max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # if name != "superglue-record" else column_names+["answers"],
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )

    if training_args.do_test:
        if data_args.test_files is not None:
            test_datasets = {test_dataset: AutoTask.get(test_dataset, test_dataset_config,
                                                        seed=data_args.data_seed).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=test_file)
                for test_dataset, test_dataset_config, test_file in zip(data_args.test_dataset_name, data_args.test_dataset_config_name, data_args.test_files)}
        else:
            test_datasets = {test_dataset: AutoTask.get(test_dataset, test_dataset_config,
                                                        seed=data_args.data_seed).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=data_args.test_file)
                for test_dataset, test_dataset_config in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)}

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)]
        for k, name in enumerate(test_datasets):
            if model_args.shared_attn is True:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(
                        preprocess_function, max_target_length=max_target_lengths[k], task_id=k),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(preprocess_function,
                                      max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )

    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    eval_metrics = [AutoTask.get(dataset_name, dataset_config_name).metric
                    for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    print(data_args.eval_dataset_name)
    compute_metrics_fn = build_compute_metrics_fn(
        data_args.eval_dataset_name, tokenizer, data_args.ignore_pad_token_for_loss) if training_args.predict_with_generate else None
    print(compute_metrics_fn)

    data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'],
                 "test": test_datasets[data_args.test_dataset_name[0]]['extra_fields'] if training_args.do_test else None,
                 "train": train_dataset['extra_fields'] if training_args.do_train else None}

    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        post_processor = AutoPostProcessor.get(data_args.dataset_name[0], tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(
            preds, labels, data_info)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    # If you want to use a different learning rate for attention layer, initialize an optimizer using the learning rate here.
    if model_args.attn_learning_rate is not None:
        all_parameters = set(model.parameters())
        attn_params = []
        for name, param in model.named_parameters():
            if name == "encoder.attn_W_up" or name == "encoder.attn_W_down" or name == "encoder.layer_norm":
                attn_params += list(param)
        attn_params = set(attn_params)
        non_attn_params = all_parameters - attn_params
        non_attn_params = list(non_attn_params)
        attn_params = list(attn_params)

        optim = AdamW([
            {'params': non_attn_params},
            {'params': attn_params, 'lr': model_args.attn_learning_rate},
        ], lr=training_args.learning_rate,)
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=training_args.warmup_steps, num_training_steps=len(
                train_dataset) * training_args.num_train_epochs // (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)
        )

        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=list(eval_datasets.values())[
                0] if training_args.do_eval else None,
            data_info=data_info,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            multi_task_compute_metrics=compute_metrics_fn,
            evaluation_metrics=TASK_TO_METRICS[data_args.dataset_name[0]],
            shared=model_args.shared_attn,
            optimizers=(optim, scheduler)
        )

    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=list(eval_datasets.values())[
                0] if training_args.do_eval else None,
            data_info=data_info,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            evaluation_metrics=TASK_TO_METRICS[data_args.dataset_name[0]],
            multi_task_compute_metrics=compute_metrics_fn,
            shared=model_args.shared_attn)

    # Saves training config.
    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_training_config(sys.argv[1], training_args.output_dir)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})

        # By setting the `save_prefix_only` True, you only save the attentions as well as the prompt components only.
        if model_args.save_prefix_only:
            save_prompts(trainer.model, output_dir=training_args.output_dir, attn_prefix_tuning=model_args.attn_prefix_tuning,
                         shared_attn=model_args.shared_attn, num_target=config.num_target, task_name=data_args.task_name)
        else:
            # save all model parameters and tokenizers regardless of whether they are updated or not.
            trainer.save_model()

        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        train_metrics["train_samples"] = min(
            max_train_samples, len(train_dataset))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        if not model_args.save_prefix_only:
            trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        trainer.save_metrics("performance", performance_metrics)

    # Validation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if model_args.attn_prefix_tuning is True:
            attention_paths = [os.path.join(training_args.output_dir, "attn_W_down.pt"), os.path.join(
                training_args.output_dir, "attn_W_up.pt")]
            trainer.model.update_attention_weights_sub(attention_paths)
            if model_args.load_layer_norm is True and "layer_norm_bias.pt" in training_args.output_dir:
                trainer.model.update_layer_norm_weights(
                    training_args.output_dir)

        if  model_args.shared_attn is False:
            for task, eval_dataset in eval_datasets.items():
                metrics = trainer.evaluate(eval_dataset=eval_dataset,
                                           max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,
                                           )
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

    # Test
    if training_args.do_test:
        logger.info("*** Test ***")
        # multi-task evaluations
        results = {}
        if model_args.shared_attn is False:
            for task, test_dataset in test_datasets.items():
                metrics = trainer.evaluate(eval_dataset=test_dataset,
                                           max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
                                           metric_key_prefix="test"
                                           )
                trainer.log_metrics("test", metrics)
                trainer.save_metrics("test", metrics)

    if model_args.save_prefix_only:
        checkpoints = glob.glob(os.path.join(
            training_args.output_dir, "checkpoint-*"))
        for checkpoint_dir in checkpoints:
            # save models
            if not os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
                continue
            checkpoint_model = torch.load(os.path.join(
                os.path.join(checkpoint_dir, "pytorch_model.bin")))
            model.load_state_dict(checkpoint_model)
            new_dir = "{}_prompt_only".format(checkpoint_dir)
            os.mkdir(new_dir)
            save_prompts(model, output_dir=new_dir, attn_prefix_tuning=model_args.attn_prefix_tuning,
                         shared_attn=model_args.shared_attn, num_target=config.num_target, task_name=data_args.task_name)

            # after saving prompts, we will remove unnecessary checkpoint dir.
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError as e:
                print("Error: %s : %s" % (checkpoint_dir, e.strerror))

    # Evaluate all checkpoints on all tasks if training_args.eval_all_at_last==True
    results = {}
    if training_args.eval_all_at_last:
        for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*_prompt_only")):
            print(checkpoint_dir)
            attention_paths = [os.path.join(checkpoint_dir, "attn_W_down.pt"), os.path.join(
                checkpoint_dir, "attn_W_up.pt")]
            trainer.model.update_attention_weights_sub(attention_paths)

            if model_args.load_layer_norm is True and "layer_norm_bias.pt" in checkpoint_dir:
                trainer.model.update_layer_norm_weights(checkpoint_dir)
            dev_metrics_all = {}
            dev_avg = []
            logger.info("*** Evaluate ***")
            for idx, (task, eval_dataset) in enumerate(eval_datasets.items()):
                if idx > 0:
                    print(task)
                    print(eval_metrics)
                shared_param = torch.load(os.path.join(
                    checkpoint_dir, "prefix_embeddings_{}.pt".format(data_args.task_name[idx])))
                trainer.model.update_prefix_weights_multi(
                    shared_param, num_target=1)
                metrics = trainer.evaluate(eval_dataset=eval_dataset,
                                           max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,
                                           )
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
                dev_metrics_all[task] = metrics
                main_metric = list(metrics.values())[0]
                dev_avg.append(main_metric)

            results.setdefault(checkpoint_dir, {})
            results[checkpoint_dir]["dev_avg"] = np.mean(dev_avg)
            results[checkpoint_dir]["dev_each"] = dev_metrics_all

        # Test
        logger.info("*** Test ***")
        for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*_prompt_only")):
            # load models here
            attention_paths = [os.path.join(checkpoint_dir, "attn_W_down.pt"), os.path.join(
                checkpoint_dir, "attn_W_up.pt")]
            trainer.model.update_attention_weights_sub(attention_paths)
            if model_args.load_layer_norm is True and "layer_norm_bias.pt" in checkpoint_dir:
                trainer.model.update_layer_norm_weights(checkpoint_dir)

            test_metrics_all = {}
            test_avg = []
            for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                shared_param = torch.load(os.path.join(
                    checkpoint_dir, "prefix_embeddings_{}.pt".format(data_args.task_name[idx])))
                trainer.model.update_prefix_weights_multi(
                    shared_param, num_target=1)
                metrics = trainer.evaluate(eval_dataset=test_dataset,
                                           max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
                                           metric_key_prefix="test"
                                           )
                trainer.log_metrics("test", metrics)
                trainer.save_metrics("test", metrics)
                test_metrics_all[task] = metrics
                main_metric = list(metrics.values())[0]
                test_avg.append(main_metric)
            results.setdefault(checkpoint_dir, {})
            results[checkpoint_dir]["test_avg"] = np.mean(test_avg)
            results[checkpoint_dir]["test_each"] = test_metrics_all
    print(results)

    return results


if __name__ == "__main__":
    main()
