from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from models import Broadcast, RelatedAnime, Studio, Picture, AnimeStatistics, Genre, AlternativeTitles, AnimeSeason, \
    AnimeStatusEnum, MediaTypeEnum


class AnimeGlobal(BaseModel):
    id: int
    title: str
    main_picture: Picture = None
    source: str = None
    rank: int
    synopsis: str
    status: AnimeStatusEnum
    created_at: datetime
    genres: list[Genre]
    my_list_status: Optional[dict]
    alternative_titles: AlternativeTitles
    start_season: AnimeSeason
    end_date: str
    num_list_users: int
    background: str
    popularity: int
    mean: float
    broadcast: Broadcast = None
    updated_at: str
    num_favorites: int
    nsfw: str
    related_anime: list[RelatedAnime]
    related_manga: list
    recommendations: list[dict]
    studios: list[Studio]
    num_scoring_users: int
    media_type: MediaTypeEnum
    rating: str
    pictures: list[dict]
    start_date: str
    average_episode_duration: int
    num_episodes: int
    statistics: AnimeStatistics

    @classmethod
    def list_fields(cls):
        return list(cls.schema().get("properties").keys())


class AnimeData(BaseModel):
    """This is used for type hinting purposes. Has no real code usefulness"""
    id: list[int]
    title: list[str]
    main_picture: list[str]  # list[Picture]
    source: list[str]
    synopsis: list[str]
    genres: list[str]  # list[list[Genre]]
    alternative_titles: list[str]  # list[AlternativeTitles]
    start_season: list[str]  # list[AnimeSeason]
    num_list_users: list[int]
    popularity: list[int]
    mean: list[float]
    broadcast: list[str]  # list[Optional[Broadcast]]
    num_favorites: list[int]
    related_anime: list[str]  # list[list[RelatedAnime]]
    studios: list[str]  # list[list[Studio]]
    num_scoring_users: list[int]
    media_type: list[str]
    rating: list[str]
    average_episode_duration: list[Optional[int]]
    num_episodes: list[int]
    statistics: list[str]  # list[AnimeStatistics]


class AnimeDataParsed(BaseModel):
    id: list[int]
    title: list[str]
    main_picture: list[Picture]
    source: list[str]
    synopsis: list[str]
    genres: list[list[Genre]]
    alternative_titles: list[AlternativeTitles]
    start_season: list[AnimeSeason]
    num_list_users: list[int]
    mean: list[float]
    broadcast: list[Optional[Broadcast]]
    num_favorites: list[int]
    related_anime: list[list[RelatedAnime]]
    studios: list[list[Studio]]
    num_scoring_users: list[int]
    media_type: list[MediaTypeEnum]
    rating: list[str]
    average_episode_duration: list[Optional[int]]
    num_episodes: list[int]
    statistics: list[AnimeStatistics]
