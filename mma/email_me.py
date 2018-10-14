import smtplib
from email.mime.text import MIMEText
from email.header import Header


mail_host = "smtp.qq.com"
mail_user = "1641315750"
mail_pass = "teshumima123"

sender = "1641315750@qq.com"
receivers = ["1641315750@qq.com"]
subject = "python smtp 邮件测试"
content = "这个是正文"
message = MIMEText(content, "plain", "utf-8")
message["from"] = sender
message["to"] = receivers[0]
message["subject"] = subject

try:
    # smtpObj = smtplib.SMTP()
    # smtpObj.connect(mail_host, 25)
    smtpObj = smtplib.SMTP_SSL(mail_host)
    smtpObj.login(mail_user, mail_pass)
    smtpObj.sendmail(sender, receivers, message.as_string())
    smtpObj.quit()
    print("邮件发送成功")
except smtplib.SMTPException as e:
    print("err:无法发送邮件", e)