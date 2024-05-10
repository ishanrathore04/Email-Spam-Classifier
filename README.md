# Spam Email Classifier
This project is a spam email classifier built using the Naive Bayes algorithm. The classifier is designed to distinguish between spam and non-spam (ham) emails based on their content. It can be used to automatically filter incoming emails and identify potentially unwanted or harmful messages.

# Features
•Naive Bayes Algorithm: The classifier is implemented using the Naive Bayes algorithm, which is a probabilistic method for classification tasks. It 
 assumes that features are conditionally independent given the class label.
 
•Text Preprocessing: The emails are preprocessed to remove stopwords, punctuation, and other irrelevant information. The text is tokenized, 
 converted to lowercase, and stemmed to reduce the dimensionality of the feature space.
 
•Vectorization: The text data is converted into numerical vectors using the CountVectorizer from scikit-learn. This step transforms the text into a 
 format suitable for machine learning algorithms.
 
•Model Training: The classifier is trained on a dataset of labeled emails, where each email is associated with a binary label indicating whether it 
 is spam or ham. The model learns to distinguish between the two classes based on the features extracted from the email content.
 
•Web Interface: The classifier is deployed as a web application using Streamlit. Users can interact with the classifier by entering custom email 
 messages or selecting from a set of sample emails. The predicted class (spam or ham) is displayed along with a confidence score.

# Dataset
The classifier is trained and evaluated on a publicly available dataset of labeled emails. The dataset contains a collection of spam and non-spam emails, each labeled with the corresponding class.

# Usage
To use the spam email classifier:
•Clone the repository to your local machine.

•Install the required dependencies using pip.

•Run the Streamlit app by executing the main Python script. Just open terminal (not python IDE) and type streamlit run [app-name].py.

•Enter a custom email message or select a sample email from the dropdown menu.

•Click the "Classify" button to see the predicted class (spam or not spam) for the email.

# Dependencies
•Python 3.x

•NumPy

•scikit-learn

•Streamlit
