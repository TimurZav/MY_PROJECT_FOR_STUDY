import glob
import os.path
import sys
import requests
import json
import smtplib
from datetime import datetime
from datetime import timedelta


filepath = sys.argv[1]


class ParseAndSendDataOnServer:

    def send_file_in_Nanonets(self, url, data):
        response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)
        json_output = json.loads(response.text)
        # with open("URL_DATA.json", "w") as write_file:
        #     json.dump(json_output, write_file, indent=4, ensure_ascii=False)
        return json_output

    def get_data_from_Nanonets(self, json_output, file):
        file_ids = [result['id'] for result in json_output['result']]
        result_dict = {}
        for id in file_ids:
            url = 'https://app.nanonets.com/api/v2/Inferences/Model/' + model_id + '/ImageLevelInferences/' + id
            response = requests.request('GET', url,
                                        auth=requests.auth.HTTPBasicAuth(api_key, ''))
            json_output = json.loads(response.text)
            with open(f"{file}.json", "w") as write_file:
                json.dump(json_output, write_file, indent=4, ensure_ascii=False)
            status = json_output["message"]
            verification_status = 'N/A'
            if status == 'Success':
                if 'is_moderated' in json_output['result'][0].keys():
                    verification_status = json_output['result'][0]['is_moderated']
                else:
                    verification_status = 'No results found in file'

            result_dict[id] = verification_status
        return result_dict

    def get_url_data_Nanonets(self, url, data):
        redirect = "https://the.url.you.want/to/redirect/to"
        callback = "https://the.url.you.want/to/receive/a/callback/on/verification/and/exit"
        expires = str((datetime.now() + timedelta(days=7)).utcfromtimestamp(0))
        response = requests.request("POST", url, files=data, auth=requests.auth.HTTPBasicAuth(api_key, ''))
        request_file_id = json_output["result"][0]["request_file_id"]
        url = "https://preview.nanonets.com/Inferences/Model/" + model_id + "/ValidationUrl/" + request_file_id + "?redirect=" + redirect \
              + "&expires=" + expires + "&callback=" + callback
        response = requests.request("POST", url, auth=requests.auth.HTTPBasicAuth(api_key, ''))
        return response.text

    def status_in_data(self, file):
        with open(f"{file}.json") as f:
            json_output_for_status = json.load(f)
        return [
            json_output_for_status['result'][0]['prediction'][i].get(
                "validation_status"
            )
            for i in range(len(json_output_for_status['result'][0]['prediction']))
        ]

    def send_url_on_server(self, url, data):
        url_nanonets = self.get_url_data_Nanonets(url, data)
        print(self.send_email("timurzavalov@gmail.com", "Timur1512", "timurka.zavyalov@mail.ru",
                              "Result OCR of Nanonets", url_nanonets))
        headers = {
            'Content-Type': 'application/json',
        }
        data_for_link = {'data_json': url_nanonets}
        data_for_link = json.dumps(data_for_link)
        return data_for_link
        # response = requests.post('http://127.0.0.1:5000/webhook', headers=headers, data=data_for_link)
        # return "Status: ", response.status_code

    def send_email(self, user, pwd, recipient, subject, body):
        FROM = user
        TO = recipient if isinstance(recipient, list) else [recipient]
        SUBJECT = subject
        TEXT = body

        message = """From: %s\nTo: %s\nSubject: %s\n\n%s
        """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
        try:
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            server.login(user, pwd)
            server.sendmail(FROM, TO, message)
            server.close()
            return 'successfully send the mail'
        except:
            return "failed to send mail"


parse_and_send_data_on_server = ParseAndSendDataOnServer()

dict_json_output = dict()

# model_id = '51fbbbb8-2d31-4c09-bbaf-6e470ea52c05'
# api_key = 'VyO_74wVEHtoE5tezhJEuQTcyvnj9kmh'

model_id = 'e5948316-75eb-4079-89b6-56c8ba35b794'
api_key = '0YsRn_wh6Zh2kizmJipRFgCFvqupG3w1'

url_Nanonets = 'https://app.nanonets.com/api/v2/OCR/Model/' + model_id + '/LabelFile/?async=true'
data_json = {'file': open(filepath, 'rb')}
json_output = parse_and_send_data_on_server.send_file_in_Nanonets(url_Nanonets, data_json)
# print(json_output)
dict_json_output[os.path.basename(filepath)] = json_output

# for json_output in dict_json_output.values():
status_validate = list(parse_and_send_data_on_server.get_data_from_Nanonets(json_output, filepath).values())[0]
print(f"Status validation image {filepath} is {status_validate}")
status = parse_and_send_data_on_server.status_in_data(filepath)
if status_validate == 'N/A':
    while list(parse_and_send_data_on_server.get_data_from_Nanonets(json_output, filepath).values())[0] == 'N/A':
        print(parse_and_send_data_on_server.send_url_on_server(url_Nanonets, data_json))
else:
    print(parse_and_send_data_on_server.send_url_on_server(url_Nanonets, data_json))

