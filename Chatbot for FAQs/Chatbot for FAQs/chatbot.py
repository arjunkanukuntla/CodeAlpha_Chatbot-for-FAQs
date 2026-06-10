import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===========================================
# FAQ DATA
# ===========================================

faq_data = {
    "What are the college timings?":
        "The college timings are from 9:00 AM to 4:00 PM.",

    "Where is the admission office?":
        "The admission office is located in Block A.",

    "How can I pay fees?":
        "Fees can be paid online or at the accounts office.",

    "Is hostel available?":
        "Yes, hostel facilities are available for both boys and girls.",

    "How do I apply for admission?":
        "You can apply online through the admission portal or visit the admission office.",

    "What courses are offered?":
        "The college offers Engineering, MBA, MCA and Degree programs.",

    "When does the semester start?":
        "The semester starts every year in July.",

    "Is there a library?":
        "Yes, the college has a central library open from 8 AM to 8 PM.",

    "Do you provide scholarships?":
        "Yes, scholarships are available for eligible students.",

    "How can I contact the college?":
        "You can contact the college through phone, email or by visiting the campus."
}

# ===========================================
# NLP
# ===========================================

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text


questions = list(faq_data.keys())
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)


def chatbot_response(user_input):

    user_input = preprocess(user_input)

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_index = similarity.argmax()

    score = similarity[0][best_index]

    if score < 0.2:
        return (
            "Sorry, I couldn't find a matching FAQ.\n"
            "Please try asking in a different way."
        )

    return faq_data[questions[best_index]]


# ===========================================
# SEND MESSAGE
# ===========================================

def send_message(event=None):

    message = entry.get().strip()

    if message == "":
        return

    chat_area.config(state="normal")

    chat_area.insert(
        tk.END,
        "👤 You:\n",
        "user_name"
    )

    chat_area.insert(
        tk.END,
        message + "\n\n",
        "user_text"
    )

    reply = chatbot_response(message)

    chat_area.insert(
        tk.END,
        "🤖 FAQ Bot:\n",
        "bot_name"
    )

    chat_area.insert(
        tk.END,
        reply + "\n\n",
        "bot_text"
    )

    chat_area.config(state="disabled")

    chat_area.see(tk.END)

    entry.delete(0, tk.END)


# ===========================================
# WINDOW
# ===========================================

root = tk.Tk()

root.title("🎓 College FAQ Chatbot")

root.geometry("900x650")

root.configure(bg="#15181f")

# ===========================================
# TITLE
# ===========================================

title = tk.Label(
    root,
    text="🎓 College FAQ Chatbot",
    bg="#15181f",
    fg="#4fc3f7",
    font=("Segoe UI", 24, "bold")
)

title.pack(pady=(15, 5))

subtitle = tk.Label(
    root,
    text="CodeAlpha Artificial Intelligence Internship - Task 2",
    bg="#15181f",
    fg="white",
    font=("Segoe UI", 11)
)

subtitle.pack()

# ===========================================
# CHAT AREA
# ===========================================

chat_area = ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Segoe UI", 12),
    bg="#20242d",
    fg="white",
    relief="flat",
    padx=12,
    pady=12
)

chat_area.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=20
)

chat_area.tag_config(
    "user_name",
    foreground="#00e676",
    font=("Segoe UI", 12, "bold")
)

chat_area.tag_config(
    "bot_name",
    foreground="#4fc3f7",
    font=("Segoe UI", 12, "bold")
)

chat_area.tag_config(
    "user_text",
    foreground="white",
    lmargin1=20,
    lmargin2=20
)

chat_area.tag_config(
    "bot_text",
    foreground="#dddddd",
    lmargin1=20,
    lmargin2=20
)

chat_area.insert(
    tk.END,
    "🤖 Welcome!\n\n",
    "bot_name"
)

chat_area.insert(
    tk.END,
    "Ask me questions about admissions, fees, hostel, scholarships, library, timings and more.\n\n",
    "bot_text"
)

chat_area.config(state="disabled")

# ===========================================
# INPUT FRAME
# ===========================================

bottom = tk.Frame(
    root,
    bg="#15181f"
)

bottom.pack(
    fill="x",
    padx=15,
    pady=(0, 15)
)

entry = tk.Entry(
    bottom,
    font=("Segoe UI", 13),
    bg="#20242d",
    fg="white",
    insertbackground="white",
    relief="flat"
)

entry.pack(
    side="left",
    fill="x",
    expand=True,
    ipady=10,
    padx=(0, 10)
)

entry.bind("<Return>", send_message)

send_button = tk.Button(
    bottom,
    text="Send ➜",
    font=("Segoe UI", 12, "bold"),
    bg="#2196f3",
    fg="white",
    activebackground="#1976d2",
    activeforeground="white",
    relief="flat",
    command=send_message,
    padx=20,
    pady=8
)

send_button.pack(side="right")

root.mainloop()