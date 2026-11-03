"""Typed data about repositories and issues."""

from pydantic import BaseModel, Field


class RepoRef(BaseModel):
    """A repository address. Both parts were already validated by urls.py."""

    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"


class IssueRef(BaseModel):
    repo: RepoRef
    number: int = Field(ge=1)


class RepoInfo(BaseModel):
    full_name: str
    description: str | None = None
    default_branch: str = "main"
    language: str | None = None
    size_kb: int = 0
    private: bool = False
    stars: int = 0
    html_url: str = ""


class IssueData(BaseModel):
    number: int
    title: str
    body: str = ""
    state: str = "open"
    labels: list[str] = Field(default_factory=list)
    author: str = ""
    html_url: str = ""