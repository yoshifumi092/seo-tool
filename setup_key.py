key = input("Groq APIキーを貼り付けてEnter: ")
with open("/Users/sai/Documents/my-first-project/seo-tool/.env", "w") as f:
    f.write("GROQ_API_KEY=" + key + "\n")
print("保存しました！")
