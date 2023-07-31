import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import shutil
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

def send_email_message(gmail, app_password, subject, body, attach_dir=None):
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
        print('server set up')

        # Send the email
        if attach_dir != None: 
            # Create a ZIP file from the directory
            zip_name = os.path.join(attach_dir, 'results')
            shutil.make_archive(zip_name, 'zip', attach_dir)
            print('zip created')
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(open(f'{zip_name}.zip', 'rb').read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{zip_name}.zip"')
            message.attach(part)
            print('zip attached')
        
        server.send_message(message)
        print('message sent')
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        # Print any error messages
        print(e)