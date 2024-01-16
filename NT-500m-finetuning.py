# Imports
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob2

# use gpu
device = torch.device("cuda")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def tokenize_function(examples):
    outputs = tokenizer(examples["data"])
    return outputs

# list of training files
ls_expressions_files = glob2.glob('data/*.csv')

test_results = {}

# Iterate through each expression training file created using prepare_data.py
for expressions_file in ls_expressions_files:
    expression = expressions_file.split('/')[-1].split('.')[0]

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", num_labels=1, ignore_mismatched_sizes=True)
    # Move model to gpu
    model = model.to(device)

    # Load expression training data
    df = pd.read_csv(expressions_file)

    # Get training data i.e. 80% of the data (len = 670 individuals)
    train_sequences = df.iloc[:536, 1].values
    train_labels = df.iloc[:536, 0].values

    # Split the dataset into a training and a validation dataset
    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(train_sequences,
                                                                                train_labels, test_size=0.1, random_state=30)

    # Get test data i.e. rest of the data
    test_sequences = df.iloc[536:, 1].values
    test_labels = df.iloc[536:, 0].values

    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")

    # Promoter dataset
    ds_train = Dataset.from_dict({"data": train_sequences,'labels':train_labels})
    ds_validation = Dataset.from_dict({"data": validation_sequences,'labels':validation_labels})
    ds_test = Dataset.from_dict({"data": test_sequences,'labels':test_labels})

    # Creating tokenized dataset sequences
    tokenized_datasets_train = ds_train.map(
        tokenize_function,
        batched=True,
        remove_columns=["data"],
    )
    tokenized_datasets_validation = ds_validation.map(
        tokenize_function,
        batched=True,
        remove_columns=["data"],
    )
    tokenized_datasets_test = ds_test.map(
        tokenize_function,
        batched=True,
        remove_columns=["data"],
    )

    # hyper-parameters
    batch_size = 8
    model_name='nucleotide-transformer'
    args_expression = TrainingArguments(
        f"{model_name}-finetuned-NucleotideTransformer",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps= 1,
        per_device_eval_batch_size= 64,
        num_train_epochs= 2,
        logging_steps= 100,
        load_best_model_at_end=True,  # Keep the best model according to the evaluation
        metric_for_best_model="rmse",
        label_names=["labels"],
        dataloader_drop_last=True,
        max_steps= 1000
    )

    # Define trainer
    trainer = Trainer(
        model.to(device),
        args_expression,
        train_dataset= tokenized_datasets_train,
        eval_dataset= tokenized_datasets_validation,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Training step (This will take some time)
    train_results = trainer.train()

    # extract metrics
    curve_evaluation_rmse =[[a['step'],a['eval_rmse']] for a in trainer.state.log_history if 'eval_rmse' in a.keys()]
    eval_rmse = [c[1] for c in curve_evaluation_rmse]
    steps = [c[0] for c in curve_evaluation_rmse]

    # Plot validation 
    plt.plot(steps, eval_rmse, 'b', label='Validation RMSE')
    plt.title('Validation rmse for gene-expression prediction for TSS: '+expression)
    plt.xlabel('Number of training steps performed')
    plt.ylabel('Validation RMSE')
    plt.legend()
    plt.savefig(expression+'-validation-rmse.png')

    # print(f"RMSE on the test dataset: {trainer.predict(tokenized_datasets_test).metrics['test_rmse']}")
    test_results[expression] = {'validation_rmse': dict(zip(steps, eval_rmse))}
    # Unload the model to be sure of not continuous finetuning for each expression
    del model
    del tokenizer
    torch.cuda.empty_cache()
