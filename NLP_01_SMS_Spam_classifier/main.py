from spam_classifier import SMS_Dataset

sms_dataset = SMS_Dataset()
sms_dataset.load_txt()

sms_dataset.preprocess_model()

