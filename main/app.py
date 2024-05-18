import streamlit as st
from LLM.helper import generate_disease_summary, generate_detailed_overview
from preprocessing import predict_image_label

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .stApp {
        padding: 10px;
    }
    .custom-column {
        padding: 0 40px;
    }
    </style>
    """, unsafe_allow_html=True
            )


def main():
    st.title("X-ray Image Classification and Disease Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="custom-column">', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Upload X-ray Image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded X-ray Image",
                     use_column_width=True, width=300)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_image is not None:
            st.markdown('<div class="custom-column">',
                        unsafe_allow_html=True)
            # Save the uploaded image to a temporary file
            predicted_label = predict_image_label(uploaded_image)

            # Predict the label using the image model
            st.subheader("Predicted Label:")
            st.write(predicted_label)
            if predicted_label == "Normal":
                st.write("You have normal condition")
            else:
                messages = generate_disease_summary(predicted_label)
                st.write(messages)

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown("<h1>Write your question?</h1>", unsafe_allow_html=True)
    follow_up_question = st.text_input(
        "Ask a follow-up question to the Language Model:")

    if follow_up_question:
        st.subheader("Detailed Overview:")

        messages = generate_detailed_overview(
            predicted_label, follow_up_question
        )

        st.write(messages)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
