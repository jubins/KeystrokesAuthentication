# Keystrokes Authentication
In this project I've built a working prototype of a system that allows software engineers to easily train, deploy and evaluate a machine learning or a neural network classifier. The system takes training data url as an input having `user_data`, `user_label` and `next` elements and analyzes their typing patterns of string **Be Authentic. Be Yourself. Be Typing.** and creates a model that can be used for prediction on any new dataset.

This app has been created using Python Django web framework as backend, SQLite as database and HTML, CSS, Bootstrap and jQuery for the fronend.

In this app I have created four interfaces:
1. **Analyze Training Set**: a view where a user can enter url of the training data and see what features the model will get trained on. This view takes care of pulling all the valid records and shows data in a tabular format.
2. **Train Model**: A view where user can enter training data url, select holdoutset size, and choose model type between regular machine learning model or neural network model and name of model pickle object they would like to save. This view also shows performance score comparison across all the models a user has trained so far.
3. **Evaluate Model**: This view allows user to choose a model and performs prediction on the test set.
4. **Homepage**: To navigate between these views.

### Setup
You will need to Python 3, Django 2.2 and SQLite to run this project.
All other dependencies are mentioned in the `requirements.txt` file or you can use directly use `venv` file as your virtual environment that has dependencies preloaded.
Once the above steps are done, go to your terminal and type
```
# To activate the virtual environment
source venv/bin/activate

# To start the django server
python manage.py runserver
```
You should see something like below output in your terminal which indicates django server has started
```
Performing system checks...

System check identified no issues (0 silenced).
March 09, 2020 - 00:51:23
Django version 3.0.4, using settings 'KeystrokesAuthentication.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```
Go to http://127.0.0.1:8000/ browser and you should be able to access the pages.

### Homepage
This is the first page user will see when they start the project
![Homepage](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/homepage.png)

### Analyze Training Set
The analyze dataset page follows these properties:
1. **Training Data Url**: Url of the training data.
2. **Display First N records**: To display top n samples of the imported dataset. Since the dataset can be huge the limit of records one can view is 1000. 

The view consists of:
1. **Data pipeline**: Takes care of parsing raw json data and preparing it in a format consumable by model for training.
2. **Feature engineering**: Generates new features like `year`, `month`, `day`, `hour`, `minute`, `second`, `microsecond` and transforms categorical features like `Be Authentic. Be Yourself. Be Typing.` into numerical.

Analyze Training Data
![AnalyzeTrainingData](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/analyze_training_data.png)

### Model Training
The Training view follows below properties:
1. **Training Data Url**: In entering training data url, if user has entered training data in analyze view it will not trigger redownloading the training data. Training data must follow this sample [expected format.](https://challenges.unify.id/v1/mle/user_4a438fdede4e11e9b986acde48001122.json)
2. **Holdout set size**: Holdout set size is between `0` to `1` to evaluate the model performance.
3. **Model Type**: User has an option to train model either a Adaboost Classifier or Recurrent Neural Networks LSTM classifier. More options can be easily added and in the backend.
4. **Model Pickle Filename**: A user input to save model for future use like prediction or performance comparison between different versions of the model.

This view consists of:
1. **Data Pipeline and Feature Extractors**: Incase user has not analyzed the dataset from Analyze page, this step takes care of data preparation for model training.
2. **Model Options**: Different model options like Adaboost Classifier and RNN LSTM Classifier. I choose Adaboost because it is less susceptible to overfitting problems although it can be sensitive to noisy data. I chose LSTMs because we are training the model on a sequential data.
3. **Saving model object and loading**: Saving the model as pickle object with specified name in the interface.
4. **Evaluation**: Evaluation of holdout set as per specified size across all the models.

Train Model Form - to enter training data url, choose holdoutset, model type and pickle file name.
![TrainModelForm](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/train_model_1.png)

Train Model Training - notifies the user once the training has started.
![TrainModelTraining](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/train_model_2.png)

Train Model Holdout Evaluation - evaluates holdout set performance across current and all the past models.
![TrainModelHoldoutEvaluation](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/train_model_3.png)

### Model Prediction
The Prediction view follows below properties:
1. **Test Data Url**: This step prepares the test data through the same data pipeline and feature extractors as mentioned earlier. Test data must follow this sample [expected format.](https://challenges.unify.id/v1/mle/sample_test.json)
2. **Choose Model**: All the models trained by the user will appear here automatically so a user can select the model they want to use for prediction.
3. **Display First N Rows**: Since the test size can be huge the limit of viewing test set predicitions is 1000.

This view consists of:
1. **View Predictions**: The chosen model gets loaded as the user clicks on this button and prediction is performed.
2. **Export Predictions To CSV**: View only displays first n predictions upto 1000. To view all the predictions a CSV file named `output_predicitions.csv` is prepared as the user clicks on this button and prompted to download it.

Predict Model Form - inputs to enter test data url, choose already trained model.
![PredictModelForm](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/evaluate_model_0.png)

Predict Model Loading Notification - notifies the user once the model is being loaded.
![PredictModelNotification](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/evaluate_model_1.png)

Predict Model Prediction - shows predictions for each user for first n users.
![PredictModelPrediction](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/evaluate_model_2.png)

Predict Model CSV Export/Download - exports the csv if user wants to view all the predictions.
![PredictModelExport](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/evaluate_model_3.png)

Predict Model Prediction CSV
![PredictModelPredictionCSV](https://github.com/jubins/KeystrokesAuthentication/blob/master/images/evaluate_model_4.png)

### Additional Questions
1. If you had one additional day, what would you change or improve to your submission?
- I would improve my model performace. In this project I have mostly focused on building an infrastructure used by software engineers that allows for the model to be easily trained, deployed and evaluated and compared across previous versions of models. If I had more time I wold also like to add some charts to visualize performance and comparison, more model options, feature engineering options to further improve my app.
2. How would you modify your solution if the number of users was 1,000 times larger?
- My current app is monolithic, I have tried separating the services for data pipeline, model training and evaluation but they still depend on each other. If the users was 1,000 times larger I would also use something like Apache Spark or distributed infrastructure to prepare training data and train the models.
3. What insights and takeaways do you have on the distribution of user performance?
- I noticed that the number of user inputs reduced as we iterate through the list of users. Also I plotted the user typing chart below if we consider the typing pattern, time (speed) or maybe even account for mistakes they make we can get a unique figerprint considering all these features.
4. What aspect(s) would you change about this challenge?
- I enjoyed working on this challenge. I wish I had more time to finish it.
5. What aspect(s) did you enjoy about this challenge?
- I enjoyed everything from exploring the dataset, model building, designing the platform to train, deploy and evaluate model.
