import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email_message(gmail, app_password, subject, body):
# Set up the SMTP server
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    # Define your credentials
    email = gmail
    # app password
    password = app_password
    # Set up the email
    message = MIMEMultipart()
    message["From"] = email
    message["To"] = email  # sending to yourself
    message["Subject"] = subject

    # Add the message body
    body = body
    message.attach(MIMEText(body, 'plain'))

    try:
        # Create a secure SSL context
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo()  # Can be omitted
        server.starttls()  # Secure the connection
        server.login(email, password)
        text = message.as_string()
        # Send the email
        server.sendmail(email, email, text)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        # Print any error messages
        print(e)