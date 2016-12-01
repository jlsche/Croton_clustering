import sys, os, inspect
from flask import Flask, request, send_from_directory
from flask_restful import Resource, Api, reqparse
import clustering, merge_cluster
import clustering_multiprocess, merge_multiprocess

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

app = Flask(__name__, static_folder='/home/clusteringing/data/')
api = Api(app)
#output_filename = 'cluster_result.csv'
'''
def sendResultMail(msg):
    sendmail_location = "/usr/sbin/sendmail" # sendmail location
    p = os.popen("%s -t" % sendmail_location, "w")
    p.write("From: %s\n" % "cljhsu@gmail.com")
    p.write("To: %s\n" % "claude.shen@lingtelli.com")
    p.write("Subject: Result of Croton Clustering\n")
    p.write("\n") # blank line separating headers from body
    p.write(msg)
    status = p.close()
    if status != 0:
        print("Sendmail exit status", status)
'''


class Task(Resource):
    def get(self, command):
        msg = command
        parser = reqparse.RequestParser()
        parser.add_argument('id')
        args = parser.parse_args()
        if command == 'start':
            dir_id = args['id']
            clustering_resp = clustering_multiprocess.main(dir_id)
            #clustering_resp = clustering.main(dir_id)
            if clustering_resp == 'No objects to concatenate.':
                return 'No objects to concatenate. (數量大於3之群才會進行分群)'
            if clustering_resp != 'OK':
                #sendResultMail('failed clustering.')
                return 'failed clustering.'
            else:
                print('finished clustering...')

            merging_resp = merge_multiprocess.merge_all_df(dir_id)
            #merging_resp = merge_cluster.merge_all_df(dir_id)
            # 是否要覆蓋原本的檔案？ 還有script_final.py的產出是否也要覆蓋？
            if merging_resp == 'OK':
                #sendResultMail('Task finished.')
                return 'Task finished.'
            else:
                #sendResultMail('failed merging.')
                return 'failed merging.'
            #return send_from_directory(app.static_folder, output_filename)

api.add_resource(Task, '/<string:command>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
