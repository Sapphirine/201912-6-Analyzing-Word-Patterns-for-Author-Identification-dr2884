import requests
from bs4 import BeautifulSoup
import json

INDEX_JSON_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/nytData/index.json'
RAW_DATA_DIR_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/nytData/rawData'

API_KEY = 'r1idEBBPtXkKydoBuqtvN8Wks2jiLo0Q'
COOKIE = 'nyt-a=AhPGl6SIfNuzxTy3YaXOWJ; nyt-gdpr=0; nyt-purr=cfs; nyt-geo=US; optimizelyEndUserId=oeu1576219851670r0.9377363992866923; edu_cig_opt=%7B%22isEduUser%22%3Afalse%7D; b2b_cig_opt=%7B%22isCorpUser%22%3Afalse%7D; _gcl_au=1.1.321250240.1576219854; __gads=ID=ffc27c59a89d4683:T=1576219854:S=ALNI_MZBanosI9wjQiF5lrkbkT0QUFCujg; walley=GA1.2.354844274.1576219854; walley_gid=GA1.2.1371404090.1576219856; nyt-us=1; iter_id=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhaWQiOiI1ZGYzMzRkNDNmODZkODAwMDFhNTBlZmIiLCJjb21wYW55X2lkIjoiNWMwOThiM2QxNjU0YzEwMDAxMmM2OGY5IiwiaWF0IjoxNTc2MjE5ODYwfQ.23AK7Wj0YuklR2Cog4KjuURBssNmgk0ELEPH8-8-4lY; _ga=GA1.2.1928005506.1576219987; _gid=GA1.2.37574780.1576219987; nyt-d=101.000000000NAI00000YAIWo0%2CPN5%2F%406366e282%2F1792a326; nyt-auth-method=sso; LPVID=I2MmIzYmQ2MWIwYTQ4NTNm; LPSID-17743901=Ilfx02-hTcaYr-0TCLP5nw; _fbp=fb.1.1576234938535.1207420482; _derived_epik=dj0yJnU9a21SSGhZel9RcndmWTQ5cGh0Q3pkRFhVTXdhdXVYX2kmbj1JcE5OSEpvbS1jUkFsamRWanliZ0V3Jm09NyZ0PUFBQUFBRjN6Y3JJ; NYT-MPS=c2e8714e313020434c26f1300c246e8fb0f16c64802e52f90813c60b3e6abaceeaed578f98568535ad91ce7036372b9b; NYT-S=1wdJAcVF9QIsm9hQdVoH3W29C8rjRjbC9.KegTG4oDLXw3N.HnxUn2VTfB2.wwjFJwjOoea6bgYnQKCTZP5x4liKkeQI7PGERoe2hZqDu7n/AgftmTNjoz4jxDTQO38FS1; datadome=1lQEtgI6NOr.fQwU9.0J9z_D~mH~d6_0mZg2hL9slABZO8Bc-4GCc6Vo1wHDLaWlLsiP90TUm-Z_rZ3HQq6ToDJwV~vKXZRyLM-8ghY_5T; nyt-jkidd=uid=104055875&lastRequest=1576235756266&activeDays=%5B0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C0%2C1%5D&adv=1&a7dv=1&a14dv=1&a21dv=1&lastKnownType=regi; purr-pref-regi=<C_; _gat_UA-58630905-2=1; nyt-m=34971D3743D8DB25EF3B67B2A47E1CD9&igf=i.0&ird=i.0&v=i.2&ft=i.0&iub=i.0&iga=i.0&iru=i.0&s=s.core&rc=i.0&pr=l.4.0.0.0.0&fv=i.0&igu=i.1&t=i.10&n=i.2&cav=i.0&iir=i.0&l=l.2.698877804.1648308549&ica=i.0&ifv=i.0&uuid=s.d3e7ef2b-ed43-4aa5-b119-3c0b46d30474&imu=i.1&prt=i.0&vp=i.0&iue=i.0&g=i.0&er=i.1576235766&ier=i.0&igd=i.0&vr=l.4.0.0.0.0&ira=i.0&e=i.1577836800&imv=i.0'

AUTHOR_LIST = ['Charlie Savage',
               'Katie Benner',
               'Christine Hauser',
               'Timothy Williams',
               'Nicholas Fandos',
               'Christine Hauser',
               'Peter Baker',
               'Eileen Sullivan',
               'James Butler',
               'Michelle Goldberg',
               'Kara Swisher',
               'David Brooks',
               'Paul Krugman',
               'Caroline Fredrickson',
               'Spencer Bokat-Lindell'
              ]

def getArticlesByAuthor(author, apiKey, cookie):
    originalString = 'By {}'.format(author)
    data = {'q': originalString,
            'api-key': apiKey}
    response = requests.get('https://api.nytimes.com/svc/search/v2/articlesearch.json',
                            params=data,
                            cookies={'cookie': cookie})

    docList = response.json()['response']['docs']
    filteredList = [d for d in docList if d['byline']['original'] == originalString]
    urlList = [doc['web_url'] for doc in filteredList]
    return urlList

def getArticleText(url, cookie):
    req = requests.get(url, cookies = {'cookie': cookie})
    soup = BeautifulSoup(req.text, 'lxml')
    return "\n".join([p.text for p in soup.find_all('p', class_= 'css-exrw3m evys1bk0')])

def getAllArticlesForAuthor(author, apiKey, cookie):
    urlList = getArticlesByAuthor(author, apiKey, cookie)
    textList = [getArticleText(url, cookie) for url in urlList]
    processedTextList = [text.replace(u'\u2018', u"'")
                .replace(u'\u2019', u"'")
                .replace(u'\u201c', u'"')
                .replace(u'\u201d', u'"')
                .replace(u'\u2014d', u'--')
                for text in textList]
    return list(zip(urlList, processedTextList))

def saveArticleDict(articleDict, name):
    with open(name, 'w') as fp:
        json.dump(articleDict, fp)

def createArticleDict(apiKey, cookie):
    articleDict = {}
    count = 1
    for author in AUTHOR_LIST:
        articleDict[author] = []
        articleList = getAllArticlesForAuthor(author, API_KEY, COOKIE)
        for (url, text) in articleList:
            articleDict[author].append({'idx': count, 'url': url, 'text': text})
            count += 1

    saveArticleDict(articleDict, INDEX_JSON_PATH)
    return articleDict