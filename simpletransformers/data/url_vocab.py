import json
import numpy as np


class UrlVocab:
    """A vocabulary of entities (articles identifiable by URL) that are recommendation candidates."""
    def __init__(self, df_url):
        self.df_url = df_url
        self.url_vocab_list = list(self.df_url['url'])
        self.vocab_size = len(self.url_vocab_list)
        self.url_to_idx = {url: i for i, url in enumerate(self.url_vocab_list)}
        self.url_to_context = {x['url']: x
                               for x in json.loads(self.df_url.to_json(orient='records'))}

        self.url_to_nav_urls = None
        self.nav_url_list = None
        self.out_connectivity, self.in_connectivity = None, None
        self._find_url_connectivity()

    def _find_url_connectivity(self):
        """Find related entities through hyperlinks"""
        # Find hierarchical edges
        self.nav_url_list = sorted(list(set(sum(self.df_url['nav_hrefs'], []))))
        self.nav_url_to_idx = {url: i for i, url in enumerate(self.nav_url_list)}
        self.url_to_nav_urls = {x: y for x, y in zip(self.df_url['url'], self.df_url['nav_hrefs'])}

        # Find cross edges
        out_url = self.df_url['hrefs']
        connectivity = []
        for ou in out_url:
            conn = [0 for _ in range(self.vocab_size)]
            for u in ou:
                if u in self.url_to_idx:
                    conn[self.url_to_idx[u]] = 1
            connectivity.append(conn)
        self.out_connectivity = np.array(connectivity)  # Cross edge adjacent matrix
        self.in_connectivity = np.transpose(self.out_connectivity)
        print(f'Number of articles with OUT hyperlinks: {sum(self.out_connectivity.sum(1) > 0)}')
        print(f'Number of articles with IN hyperlinks:  {sum(self.in_connectivity.sum(1) > 0)}')
        print(f'Number of articles with OUT or IN hyperlinks:',
              f'{sum((self.out_connectivity.sum(1) > 0) | (self.in_connectivity.sum(1) > 0))}')

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

    def url2outconn(self, url):
        return self.out_connectivity[self.url2idx(url)].tolist()

    def url2inconn(self, url):
        return self.in_connectivity[self.url2idx(url)].tolist()

    def url2nav(self, url):
        """Return a binary representation of the navigation nodes of a URL."""
        return self.encode_nav_urls(self.url_to_nav_urls[url])

    def encode_nav_urls(self, nav_urls):
        if self.nav_url_list is None:
            return None
        return [1 if x in nav_urls else 0
                for x in self.nav_url_list]