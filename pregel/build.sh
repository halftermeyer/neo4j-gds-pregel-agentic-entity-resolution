#!/usr/bin/env bash
# Build a custom GDS Pregel procedure and deploy it to a self-managed Neo4j.
# Usage: DBMS_HOME=/path/to/dbms SRC=src/main/java NAME=my-pregel ./build.sh [--deploy]
set -euo pipefail
: "${DBMS_HOME:?set DBMS_HOME to the Neo4j home (contains plugins/, lib/, conf/)}"
SRC="${SRC:-src/main/java}"
NAME="${NAME:-custom-pregel}"
GDS_JAR=$(ls "$DBMS_HOME"/plugins/neo4j-graph-data-science-*.jar | head -1)

rm -rf build && mkdir -p build
# -proc:none: do not run annotation processors bundled in the GDS jar
javac -proc:none --release 21 -cp "$GDS_JAR:$DBMS_HOME/lib/*" -d build $(find "$SRC" -name '*.java')
jar cf "$NAME.jar" -C build .
echo "built $NAME.jar"

if [[ "${1:-}" == "--deploy" ]]; then
  cp "$NAME.jar" "$DBMS_HOME/plugins/"
  echo "copied to $DBMS_HOME/plugins/"
  echo "Make sure conf/neo4j.conf allows your namespace in BOTH:"
  echo "  dbms.security.procedures.unrestricted=...,myns.*"
  echo "  dbms.security.procedures.allowlist=...,myns.*"
  echo "Then restart (Neo4j Desktop: JAVA_HOME=<Desktop bundled JRE> \$DBMS_HOME/bin/neo4j restart)"
  echo "If the procedure is missing after restart: grep -i 'IllegalAccess\|myns' \$DBMS_HOME/logs/debug.log"
fi
