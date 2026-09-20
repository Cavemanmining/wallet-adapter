"""The bots Jarvis ships with.

One module per bot, each a :class:`~jarvis_bots.contracts.Bot`.  Nothing is
imported here: :mod:`jarvis_bots.registry` says "explicit registration only
... no entry-point scan, no import of every module in a package", and a
package ``__init__`` that imported its siblings would be exactly the
implicit discovery that file refuses.  A composition root imports the one
bot it wants::

    from jarvis_bots.bots.poke_bot import PokeBot

so which bots run is a wiring line you can read, not a directory listing.
"""

from __future__ import annotations

__all__: list = []
