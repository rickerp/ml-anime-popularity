from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, AnyHttpUrl


class ReverseEnum(Enum):
    @classmethod
    def max(cls): return max([m.value for m in cls.__members__.values()])

    @classmethod
    def min(cls): return min([m.value for m in cls.__members__.values()])

    @classmethod
    def _missing_(cls, value: str):
        if isinstance(value, float) and np.isnan(value):
            return cls["nan"]
        return cls[value] if value in cls.__members__ else None


class AnimeSeasonEnum(ReverseEnum):
    nan = 0
    winter = 1
    spring = 2
    summer = 3
    fall = 4


RatingEnum = ReverseEnum("RatingEnum", "nan g pg pg_13 r r+ rx")
MediaTypeEnum = ReverseEnum("MediaTypeEnum", "unknown movie music ona ova special tv")
SourceEnum = ReverseEnum("SourceEnum", "nan 4_koma_manga book card_game game light_novel manga mixed_media music novel "
                                       "original other picture_book radio visual_novel web_manga")


class Picture(BaseModel):
    medium: AnyHttpUrl
    large: AnyHttpUrl


class Genre(BaseModel):
    id: int
    name: str
    # ['Action', 'Adult Cast', 'Adventure', 'Anthropomorphic', 'Avant Garde', 'Award Winning', 'Boys Love', 'CGDCT', 'Childcare', 'Combat Sports', 'Comedy', 'Crossdressing', 'Delinquents', 'Detective', 'Drama', 'Ecchi', 'Educational', 'Erotica', 'Fantasy', 'Gag Humor', 'Girls Love', 'Gore', 'Gourmet', 'Harem', 'Hentai', 'High Stakes Game', 'Historical', 'Horror', 'Idols (Female)', 'Idols (Male)', 'Isekai', 'Iyashikei', 'Josei', 'Kids', 'Love Polygon', 'Magical Sex Shift', 'Mahou Shoujo', 'Martial Arts', 'Mecha', 'Medical', 'Military', 'Music', 'Mystery', 'Mythology', 'Organized Crime', 'Otaku Culture', 'Parody', 'Performing Arts', 'Pets', 'Psychological', 'Racing', 'Reincarnation', 'Reverse Harem', 'Romance', 'Romantic Subtext', 'Samurai', 'School', 'Sci-Fi', 'Seinen', 'Shoujo', 'Shounen', 'Showbiz', 'Slice of Life', 'Space', 'Sports', 'Strategy Game', 'Super Power', 'Supernatural', 'Survival', 'Suspense', 'Team Sports', 'Time Travel', 'Vampire', 'Video Game', 'Visual Arts', 'Workplace']


class AnimeSeason(BaseModel):
    year: Optional[int] = None
    season: AnimeSeasonEnum = AnimeSeasonEnum.nan


class AnimeStatusEnum(Enum):
    finished_airing = "finished_airing"
    currently_airing = "currently_airing"
    not_yet_aired = "not_yet_aired"


class AlternativeTitles(BaseModel):
    synonyms: list[str]
    en: str
    ja: str


class Broadcast(BaseModel):
    day_of_the_week: str = None
    start_time: str = None


class Studio(BaseModel):
    id: int
    name: str


class AnimeStatistics(BaseModel):
    class Status(BaseModel):
        watching: str
        completed: str
        on_hold: str
        dropped: str
        plan_to_watch: str

    status: Status
    num_list_users: int


class RelatedAnime(BaseModel):
    class AnimeNode(BaseModel):
        id: int
        title: str
        main_picture: Picture = None

    node: AnimeNode
    relation_type: str
    relation_type_formatted: Optional[str]


class Anime(BaseModel):
    id: int
    title: str
    main_picture: Picture = None
    source: SourceEnum = SourceEnum["nan"]
    synopsis: str = ""
    genres: list[Genre] = []
    alternative_titles: AlternativeTitles
    start_season: AnimeSeason = AnimeSeason()
    num_list_users: int
    popularity: int
    mean: float
    broadcast: Broadcast = None
    num_favorites: int
    related_anime: list[RelatedAnime] = []
    studios: list[Studio]
    num_scoring_users: int
    media_type: MediaTypeEnum
    rating: RatingEnum = RatingEnum["nan"]
    average_episode_duration: int = None
    num_episodes: int = None
    statistics: AnimeStatistics

    @classmethod
    def list_fields(cls):
        return list(cls.schema().get("properties").keys())
