import streamlit as st
import pickle
import numpy as np

# Load the model pipeline (assuming it includes the preprocessor)
with open('C:/UB/EAS503-2/Group12_Prog_DB_project/mlartifacts/325620995994807797/e213dc0685044711a866d8191490c296/artifacts/best_estimator/model.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

def user_input_features(channel_number):
    st.subheader(f'Channel {channel_number} Input Features')
    watch_time = st.number_input(f'Insert Watch Time for Channel {channel_number} (Minutes)', value=0, key=f'watch_time_{channel_number}')
    stream_time = st.number_input(f'Insert Stream Time for Channel {channel_number} (Minutes)', value=0, key=f'stream_time_{channel_number}')
    views_gained = st.number_input(f'Insert Views Gained for Channel {channel_number}', value=0, key=f'views_gained_{channel_number}')
    channel = st.text_input(f'Channel Name for Channel {channel_number}', key=f'channel_{channel_number}')
    language = st.selectbox(f'Language for Channel {channel_number}', ['English', 'Portuguese', 'Spanish', 'Other'], key=f'language_{channel_number}')
    mature = st.selectbox(f'Is Channel {channel_number} Mature?', ['Yes', 'No'], key=f'mature_{channel_number}')
    
    return [watch_time, stream_time, views_gained, channel, language, mature]

st.title('Twitch Follower Prediction')
features_channel_1 = user_input_features(1)
features_channel_2 = user_input_features(2)

if st.button('Predict Followers for Both Channels'):
    # Stack the features for both channels into a single array for prediction
    features = np.array([features_channel_1, features_channel_2])
    predictions = model_pipeline.predict(features)
    st.write(f'Predicted Followers for Channel 1: {predictions[0]}')
    st.write(f'Predicted Followers for Channel 2: {predictions[1]}')
