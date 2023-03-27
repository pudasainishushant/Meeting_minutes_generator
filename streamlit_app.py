import nltk
nltk.download("punkt")

import streamlit as st
import os
from generate_meeting_minutes import generate_summary,generate_action_items
# Add a heading
st.title("Meeting summarizer")
# Add some content below the heading
st.write("Generate summary of the meeting")

uploaded_file = st.file_uploader("Upload a .txt file here",type="txt")
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    # Save the file in the working directory
    file_name = uploaded_file.name
    with open(os.path.join(file_name), "w") as f:
        f.write(text)
    # st.write(f"{file_name} saved in working directory!")

    st.header("Minutes of the meeting")
    
    try:
        summary = generate_summary(file_name)
        st.header("Meeting summary")
        st.write(summary)
    except:
        st.write("Sorry could not generate meeting summary")

    try:        
        actions = generate_action_items(file_name)
        st.header("Meeting Action Items")
        st.write(actions)
    except:
        st.write("Sorry could not generate meeting actions")

