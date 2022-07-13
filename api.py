import requests


class MyAnimeListAPI:
    MAL_API_URL = "https://api.myanimelist.net/v2"

    def __init__(self, client_id, anime_model=None):
        self.r = requests.session()
        self.r.headers.update({"X-MAL-CLIENT-ID": client_id})
        self.AnimeModel = anime_model

    def get_anime(self, anime_id):
        req_url = f"{self.MAL_API_URL}/anime/{anime_id}"
        if self.AnimeModel:
            req_url += '?fields=' + ','.join(self.AnimeModel.list_fields())

        res = self.r.get(req_url)
        return self.AnimeModel(**res.json()) if self.AnimeModel else res.json()
