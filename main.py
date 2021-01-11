# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import datetime
from logging import Logger
from file_processing import FileProcessor
from model_operations import Modeller
from utils import FileUtils, LoggerUtil
from sklearn.metrics import confusion_matrix, classification_report

data_path = "data"
save_model_path = "saved_models"
session_id = str(datetime.datetime.now().timestamp())
utils = FileUtils()
logger = LoggerUtil(session_id,'files')

def prepare_dataset(dir_path, is_train=True):
    processor = FileProcessor(dir_path, is_train)
    processor.process_all_files()
    return processor.get_dataset()


def prepare_model(dataset):
    modeller = Modeller(dataset)
    modeller.define_model()
    modeller.train_model()
    return modeller


def display_metrics(results_data):
    print(results_data.head())
    c_matrix = confusion_matrix(results_data["actual"], results_data["predicted"])
    print("======Confusion Matrix=====")
    print(c_matrix)
    print(classification_report(results_data["actual"], results_data["predicted"]))
    logger.log_message(results_data.head(20))
    logger.log_message(c_matrix)
    logger.log_message(classification_report(results_data["actual"], results_data["predicted"]))


if __name__ == '__main__':
    # training
    logger.log_message("start training")
    utils.update_files_list(os.path.join(data_path, "train"))
    train_data = prepare_dataset(utils.get_files())
    logger.log_message("Train data procession complete")
    modeller = prepare_model(train_data)

    # save model
    logger.log_message("Training model saved at model_"+session_id)
    modeller.get_model().save(os.path.join(save_model_path, "model_" + session_id))

    # testing
    logger.log_message("start testing")
    utils.update_files_list(os.path.join(data_path, "test"))
    test_data = prepare_dataset(utils.get_files(), False)
    logger.log_message("Test data procession complete")
    predicted_results = modeller.predict_output(test_data)

    # display metrics
    display_metrics(predicted_results)
    logger.log_message("==========Execution completed ==============")