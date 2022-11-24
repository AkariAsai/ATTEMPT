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
import functools
import logging
import json
from tabnanny import check
from pytz import common_timezones
import torch 
import os
import numpy as np

from data.tasks import TASK_MAPPING
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import sys
import subprocess
from typing import Optional, List

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils import get_adapter_config
from data import AutoTask
from data import TaskDataCollatorForSeq2Seq
from metrics.metrics import TASK_TO_METRICS
from third_party.trainers import Seq2SeqTrainer
from options import AdapterTrainingArguments, ModelArguments, DataTrainingArguments, TrainingArguments
from utils import modify_model_after_init, save_training_config 
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments 
from third_party.models import T5Config, T5ForConditionalGeneration
from data import AutoPostProcessor
import glob
import shutil


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
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    # Detecting last checkpoint.
    last_checkpoint = None

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

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
    config.shared_attn = model_args.shared_attn
    config.prefix_num = model_args.prefix_num
    config.num_target = len(data_args.task_name)
    config.ignore_target = model_args.ignore_target
    config.temperature = model_args.temperature
    config.learned_temperature = model_args.learned_temperature
    config.fix_attention = model_args.fix_attention
    adapter_config = get_adapter_config(adapter_args, data_args, training_args, config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

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
        
        if model_args.attn_prefix_tuning is True:
            model.store_prefix_weights(shared_params)

    data_args.dataset_name = data_args.task_name
    data_args.eval_dataset_name = data_args.eval_dataset_name
    data_args.test_dataset_name = data_args.test_dataset_name
    data_args.dataset_config_name = data_args.dataset_config_name
    data_args.eval_dataset_config_name = data_args.eval_dataset_config_name
    data_args.test_dataset_config_name = data_args.test_dataset_config_name
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    
    def preprocess_function(examples, max_target_length, task_id=None):
        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["extra_fields"] = examples['extra_fields']
        # if task_id is not None:
        #     model_inputs["task_ids"] = [task_id for _ in  examples['extra_fields']]

        return model_inputs 

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}

    model.resize_token_embeddings(len(tokenizer))
    model = modify_model_after_init(model, training_args, adapter_args, adapter_config)


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
    max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
        tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
        for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]
    for k, name in enumerate(eval_datasets):
        if model_args.shared_attn is True:
            eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function, max_target_length=max_target_lengths[k], task_id=k),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names, # if name != "superglue-record" else column_names+["answers"],
                    load_from_cache_file=not data_args.overwrite_cache,
            )
        else:
            eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function, max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names, # if name != "superglue-record" else column_names+["answers"],
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
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
            tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
            for dataset_name, dataset_config_name in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)]
        for k, name in enumerate(test_datasets):
            if model_args.shared_attn is True:
                test_datasets[name] = test_datasets[name].map(
                        functools.partial(preprocess_function, max_target_length=max_target_lengths[k], task_id=k),
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                )
            else:
                test_datasets[name] = test_datasets[name].map(
                        functools.partial(preprocess_function, max_target_length=max_target_lengths[k]),
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                )


    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric, we assume we have only one training task.
    eval_metrics = [AutoTask.get(dataset_name, dataset_config_name).metric\
        for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    # Extracts the extra information needed to evaluate on each dataset.
    # These information are only used in the compute_metrics.
    # We will assume that the test/eval dataloader does not change the order of 
    # the data.
    data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'],
                 "test": test_datasets[data_args.test_dataset_name[0]]['extra_fields'] if training_args.do_test else None, 
                 "train": None}

    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        post_processor = AutoPostProcessor.get(data_args.dataset_name[0], tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
        result = {}
        print(eval_metrics)
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=list(eval_datasets.values())[0] if training_args.do_eval else None,
        data_info = data_info,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        evaluation_metrics = TASK_TO_METRICS[data_args.dataset_name[0]],
        shared=model_args.shared_attn
    )
    # Saves training config. 
    if trainer.is_world_process_zero():
       os.makedirs(training_args.output_dir, exist_ok=True)
       save_training_config(sys.argv[1], training_args.output_dir)


    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)
    
    results = {}
    if training_args.do_eval:
        for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*_prompt_only")):
            print(checkpoint_dir)
            # load models here
            # TODO: add linear model loading options
            attention_paths = [os.path.join(checkpoint_dir, "attn_W_down.pt"), os.path.join(checkpoint_dir, "attn_W_up.pt")]
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
                shared_param =  torch.load(os.path.join(checkpoint_dir, "prefix_embeddings_{}.pt".format(model_args.task_id)))
                trainer.model.update_prefix_weights_multi(shared_param, num_target=1)
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
    if training_args.do_test:
        logger.info("*** Test ***")
        for checkpoint_dir in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*_prompt_only")):
            # load models here
            attention_paths = [os.path.join(checkpoint_dir, "attn_W_down.pt"), os.path.join(checkpoint_dir, "attn_W_up.pt")]
            trainer.model.update_attention_weights_sub(attention_paths)
            if model_args.load_layer_norm is True and "layer_norm_bias.pt" in checkpoint_dir:
                trainer.model.update_layer_norm_weights(checkpoint_dir)

            test_metrics_all = {}
            test_avg = []
            for idx, (task, test_dataset) in enumerate(test_datasets.items()):
                shared_param =  torch.load(os.path.join(checkpoint_dir, "prefix_embeddings_{}.pt".format(model_args.task_id)))
                trainer.model.update_prefix_weights_multi(shared_param, num_target=1)
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

    # eval final model
    attention_paths = [os.path.join(training_args.output_dir, "attn_W_down.pt"), os.path.join(training_args.output_dir, "attn_W_up.pt")]
    trainer.model.update_attention_weights_sub(attention_paths)

    dev_metrics_all = {}
    dev_avg = []

    for idx, (task, eval_dataset) in enumerate(eval_datasets.items()):
        if idx > 0:
            print(task)
            print(eval_metrics)
        shared_param =  torch.load(os.path.join(training_args.output_dir, "prefix_embeddings_{}.pt".format(model_args.task_id)))
        trainer.model.update_prefix_weights_multi(shared_param, num_target=1)
        metrics = trainer.evaluate(eval_dataset=eval_dataset,
        max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
        metric_key_prefix="eval"
        )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        dev_metrics_all[task] = metrics
        main_metric = list(metrics.values())[0]
        dev_avg.append(main_metric)

    if training_args.do_test:
        test_metrics_all = {}
        test_avg = []
        for idx, (task, test_dataset) in enumerate(test_datasets.items()):
            shared_param =  torch.load(os.path.join(training_args.output_dir, "prefix_embeddings_{}.pt".format(model_args.task_id)))
            trainer.model.update_prefix_weights_multi(shared_param, num_target=1)
            metrics = trainer.evaluate(eval_dataset=test_dataset,
            max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
            metric_key_prefix="test"
            )
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
            test_metrics_all[task] = metrics
            main_metric = list(metrics.values())[0]
            test_avg.append(main_metric)

        results.setdefault("final", {})
        results["final"]["dev_avg"] = np.mean(dev_avg)
        results["final"]["dev_each"] = dev_metrics_all
        results["final"]["test_avg"] = np.mean(test_avg)
        results["final"]["test_each"] = test_metrics_all
    
    with open(os.path.join(training_args.output_dir, "checkpoint_eval_{}.json".format(model_args.task_id)), "w") as outfile:
        json.dump(results, outfile)
    return results


if __name__ == "__main__":
    main()
