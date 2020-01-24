import json


class UrlVocab:
    def __init__(self, df_url):
        self.df_url = df_url
        self.url_vocab_list = list(df_url['url'])
        self.vocab_size = len(self.url_vocab_list)
        self.url_to_idx = {url: i for i, url in enumerate(self.url_vocab_list)}
        self.url_to_context = {x['url']: x
                               for x in json.loads(df_url.to_json(orient='records'))}

    def idx2url(self, i):
        return self.url_vocab_list[i]

    def url2idx(self, url):
        return self.url_to_idx[url]

    def encode_urls(self, urls):
        return [1 if x in urls else 0
                for x in self.url_vocab_list]

    def in_vocab(self, url):
        return url in self.url_to_idx

    def get_name(self, url):
        return self.url_to_context[url]['name']

    def get_email_context(self, url):
        return self.url_to_context[url]['context']

    def get_email_anchor(self, url):
        return self.url_to_context[url]['context']

    def get_title(self, url):
        return self.url_to_context[url]['header']

    def get_text(self, url):
        return self.url_to_context[url]['article']