from corebm.tools.base import Tool
from corebm.tools.info_database import InfoDatabase
from corebm.tools.interaction import InteractionRetriever

TOOL_MAP: dict[str, type] = {
    'info': InfoDatabase,
    'interaction': InteractionRetriever,
}