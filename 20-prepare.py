# we should run 90-last.py first to make short cut ready for nb edit, like
# %run -i C:/users/frank.wang/.ipython/profile_default/startup/99-last.py

# before run 50-middle.py that adjust nb wides and load necessary package and function we need to install:
# !python -m pip install seaborn
# !python -m pip install statsmodels
# !python -m pip install sklearn
# !python -m pip install lightgbm
# !python -m pip install shap
# !python -m pip install xgboost

# find the python version via !python --version, and download corresponding minepy version cpxxxx from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# install from the downloaded whl:
# !python -m pip install "\\path-to\minepy-1.2.5-cp37-cp37m-win_amd64.whl"

# %run -i C:/users/frank.wang/.ipython/profile_default/startup/50-middle.py

# google drive management function from api
# install pkg:
# !python -m pip install google-api-python-client
# !python -m pip install oauth2client

# from https://console.cloud.google.com/home create a project, add OAuthen credential for desktop app, download json, rename to client_secrets.json
# put in dir of !echo %cd%, add self to tester

def getcredential():
    import httplib2
    import os
    import sys
    from apiclient.discovery import build
    from oauth2client.file import Storage
    from oauth2client.client import flow_from_clientsecrets 
    from oauth2client import tools
    CLIENT_SECRETS_FILE = "client_secrets.json"
    MISSING_CLIENT_SECRETS_MESSAGE = ""
    flow = flow_from_clientsecrets(CLIENT_SECRETS_FILE,  message=MISSING_CLIENT_SECRETS_MESSAGE, scope="https://www.googleapis.com/auth/drive")
    storage = Storage("%s-oauth2.json" % sys.argv[-1])
    flags = tools.argparser.parse_args(args=[])
    creds = tools.run_flow(flow, storage, flags)
    print("Access Token: %s" % creds.access_token)
    print("Refresh Token: %s" % creds.refresh_token)
    print("Token expiry: %s" % creds.token_expiry)
    return creds
    
# v2 use title, insert, v3 use name, create
def getservice(creds):
    import httplib2
    from apiclient.discovery import build
    creds.refresh(httplib2.Http())  # refresh the access token (optional)    
    http = creds.authorize(httplib2.Http())  # apply the credentials
    http.redirect_codes = http.redirect_codes - {308}
    service = build('drive', 'v2', http=http)
    return service    

# include files in 'root'
def getallfiles1(service):
    results = service.files().list(q="mimeType='application/vnd.google-apps.folder' and trashed = false",fields="nextPageToken, items(id, title)",maxResults=40000).execute()
    allfolders=dict()
    for file in results.get('items', []):
        allfolders[file.get('id')]=file.get('title')
    all_files=[]
    response = service.files().list(q="'root' in parents and trashed=false",
                                          fields='nextPageToken, items(id, title, fileSize, modifiedDate, parents)',
                                          maxResults=40000).execute()
    for file in response.get('items', []):
        all_files.append([file.get('parents')[0].get('id'),'root',file.get('title'),file.get('id'), file.get('fileSize'),file.get('modifiedDate')])     
    for folder_id in allfolders.keys():
        response = service.files().list(q="parents in '"+folder_id+"'",
                                              fields='nextPageToken, items(id, title, fileSize, modifiedDate)',
                                              maxResults=40000).execute()
        for file in response.get('items', []):
            all_files.append([folder_id,allfolders[folder_id],file.get('title'),file.get('id'), file.get('fileSize'),file.get('modifiedDate')]) 
#     pd.DataFrame(all_files) 
    return all_files
    
# id to title
def getallfolders(service):
    results = service.files().list(q="mimeType='application/vnd.google-apps.folder' and trashed = false",fields="nextPageToken, items(id, title)",maxResults=40000).execute()
    allfolders=dict()
    for file in results.get('items', []):
        allfolders[file.get('id')]=file.get('title')
    return allfolders
    
# title to id, may be wrong for multiple files with same title
def getallfolders1(service):
    results = service.files().list(q="mimeType='application/vnd.google-apps.folder' and trashed = false",fields="nextPageToken, items(id, title)",maxResults=40000).execute()
    allfolders=dict()
    for file in results.get('items', []):
        allfolders[file.get('title')]=file.get('id')
    return allfolders

# print id with title
def uploadfolder1(service, folder, remotedirid,startfile):
    import os
    import glob
    import datetime
    from apiclient import errors
    from apiclient.http import MediaFileUpload
    files = sorted([f for f in glob.glob(folder+"/*") if os.path.getmtime(f)>=os.path.getmtime(startfile)], key=os.path.getmtime)
    for file0 in files:
        try:
            file = service.files().insert(
                body={'title': os.path.basename(file0), 'setModifiedDate' : True, 'modifiedDate' : datetime.datetime.fromtimestamp(os.path.getmtime(file0)).isoformat()+'Z', 'parents' : [{'id':remotedirid}]},
                media_body=MediaFileUpload(file0, resumable=True)).execute()
            print('%s : %s'  % (file['id'], file['title']))
        except errors.HttpError as error:
            print ('An error occurred: %s' % error)

# print id with title, and recursive to subfolder, glob only list current dir and skip .xxx
def uploadfolder2(service, folder, remotedirid,startfile):
    import os
    import glob
    import datetime
    from apiclient import errors
    from apiclient.http import MediaFileUpload
    files = sorted([f for f in glob.glob(folder+"/*") if os.path.getmtime(f)>=os.path.getmtime(startfile)], key=os.path.getmtime)
    for file0 in files:
        try:
            if os.path.isdir(file0):
                file = service.files().insert(
                    body={'title': os.path.basename(file0), 'setModifiedDate' : True, 
                          'modifiedDate' : datetime.datetime.fromtimestamp(os.path.getmtime(file0)).isoformat()+'Z', 'mimeType': 'application/vnd.google-apps.folder', 'parents' : [{'id':remotedirid}]},
                    fields='id,title').execute()
                print('%s : %s'  % (file['id'], file['title']))
                uploadfolder2(service, file0, file['id'],startfile)
            else:
                file = service.files().insert(
                    body={'title': os.path.basename(file0), 'setModifiedDate' : True, 
                          'modifiedDate' : datetime.datetime.fromtimestamp(os.path.getmtime(file0)).isoformat()+'Z', 'parents' : [{'id':remotedirid}]},
                    media_body=MediaFileUpload(file0, resumable=True)).execute()
                print('%s : %s'  % (file['id'], file['title']))
        except errors.HttpError as error:
            print ('An error occurred: %s' % error)

# print id with title, folder, modifiedtime,  and recursive to subfolder, glob only list current dir and skip .xxx
def uploadfolder3(service, folder, remotedirid, remotedir, startfile):
    import os
    import glob
    import datetime
    from apiclient import errors
    from apiclient.http import MediaFileUpload
    files = sorted([f for f in glob.glob(folder+"/*") if os.path.getmtime(f)>=os.path.getmtime(startfile)], key=os.path.getmtime)
    for file0 in files:
        try:
            if os.path.isdir(file0):
                file = service.files().insert(
                    body={'title': os.path.basename(file0), 'setModifiedDate' : True, 
                          'modifiedDate' : datetime.datetime.fromtimestamp(os.path.getmtime(file0)).isoformat()+'Z', 'mimeType': 'application/vnd.google-apps.folder', 'parents' : [{'id':remotedirid}]},
                    fields='id,title,modifiedDate').execute()
                print('%s : %s : %s : %s'  % (file['id'], file['title'], remotedir, file['modifiedDate']))
                uploadfolder3(service, file0, file['id'], file['title'],startfile)
            else:
                file = service.files().insert(
                    body={'title': os.path.basename(file0), 'setModifiedDate' : True, 
                          'modifiedDate' : datetime.datetime.fromtimestamp(os.path.getmtime(file0)).isoformat()+'Z', 'parents' : [{'id':remotedirid}]},
                    media_body=MediaFileUpload(file0, resumable=True),fields='id,title,modifiedDate').execute()
                print('%s : %s : %s : %s'  % (file['id'], file['title'], remotedir, file['modifiedDate']))
        except errors.HttpError as error:
            print ('An error occurred: %s' % error)       

# create remote folder with name given in local path file0
def createfolder(service,file0):
    import os
    import datetime
    from apiclient import errors
    try:
        file = service.files().insert(
            body={'title': os.path.basename(file0), 'setModifiedDate' : True, 'modifiedDate' : datetime.datetime.fromtimestamp(os.path.getmtime(file0)).isoformat()+'Z', 'mimeType': 'application/vnd.google-apps.folder'},
            fields='id,title').execute()
        print('%s : %s'  % (file['id'], file['title']))
    except errors.HttpError as error:
        print ('An error occurred: %s' % error)

def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv[0] if isinstance(vv, list) else vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }
             
# get info given file name
def getmetadata(service,file):
    import pandas as pd
    response = service.files().list(q="title='"+file+"'",
        spaces='drive',
        fields='nextPageToken, items(id, title, parents, modifiedDate)',
        maxResults=40000,
        orderBy='title').execute()
    return pd.DataFrame([flatten_dict(item,'.') for item in response.get('items', [])])

# get complete  file list flatterned info given file name
def getmetadata3(service,file):
    from pathlib import Path 
    import pandas as pd
    stem = Path(file).stem
    response = service.files().list(q="title contains '"+stem+"'",
        spaces='drive',
        fields='nextPageToken, items(id, title, parents, modifiedDate)',
        maxResults=40000,
        orderBy='title').execute()
    return pd.DataFrame([flatten_dict(item,'.') for item in response.get('items', [])])

     
# use pydrive pkg, which can only upload small files
%pip install pydrive
%pip install google_auth

def getdrive():
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive 
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    return drive
    
def getfiles(drive, folderid):
    fileList = drive.ListFile({'q': "parents in '"+folderid+"' and trashed=false"}).GetList()
    files=[]
    for file in fileList:
        files.append([file['title'], file['id'],file['fileSize'], file['modifiedDate']]) 
    return files
  