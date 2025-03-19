# -*- coding: utf-8 -*-

# The current tutorial is an updated version of the code available in the tutorial https://www.analyticsvidhya.com/blog/2021/12/fine-tune-bert-model-for-sentiment-analysis-in-google-colab/

from transformers import BertTokenizer
import tensorflow_datasets as tfds
from transformers import TFBertForSequenceClassification
import tensorflow as tf


# remember when we talked about the BertTokenizer
def convert_example_to_feature(review):
  return tokenizer.encode_plus(review,
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = max_length, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
              )

# can be up to 512 for BERT as discussed in class
max_length = 512

# feel free to play around with the batch size for speed vs accuracy evaluation
batch_size = 6

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label


def encode_examples(ds, limit=-1):
  # prepare list, so that we can build up final TensorFlow dataset from slices.
  input_ids_list = []
  token_type_ids_list = []
  attention_mask_list = []
  label_list = []

  if (limit > 0):
      ds = ds.take(limit)

  for review, label in tfds.as_numpy(ds):

    bert_input = convert_example_to_feature(review.decode())
    input_ids_list.append(bert_input['input_ids'])

    token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append([label])

  return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


# example of loading the imdb_reviews dataset - update that part with our amazon reviews
(ds_train, ds_test), ds_info = tfds.load('imdb_reviews', split = (tfds.Split.TRAIN, tfds.Split.TEST), as_supervised=True, with_info=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# feel free to play around with the learning rate
learning_rate = 2e-5

# Currently 1 epoch for testing, however additional epochs could potentially lead to a better result (carefull not overfit the model since we are using a pretrained BERT!)
number_of_epochs = 1

# model initialization - using uncased is used for lowercase
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# choosing Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# train dataset
ds_train_encoded = encode_examples(ds_train).shuffle(10000).batch(batch_size)

# test dataset
ds_test_encoded = encode_examples(ds_test).batch(batch_size)

# Finally, we are training our model
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)

# Example of testing, you will need to modify that part to accomodate the full test
test_sentence = "This is a really good movie. I loved it and will watch again"

# don't forget to tokenize your test inputs
predict_input = tokenizer.encode(test_sentence, truncation=True, padding=True, return_tensors="tf")

tf_output = model.predict(predict_input)[0]
tf_prediction = tf.nn.softmax(tf_output, axis=1)

labels = ['Negative','Positive'] #(0:negative, 1:positive)
label = tf.argmax(tf_prediction, axis=1)
label = label.numpy()
print(labels[label[0]])
