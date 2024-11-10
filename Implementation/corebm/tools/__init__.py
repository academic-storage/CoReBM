from Implementation.corebm import Tool
from Implementation.corebm import InfoDatabase
from Implementation.corebm import InteractionRetriever

TOOL_MAP: dict[str, type] = {
    'info': InfoDatabase,
    'interaction': InteractionRetriever,
}