import os
from pathlib import Path

from pyspark.sql import SparkSession

_HADOOP_HOME = Path(__file__).resolve().parents[1] / "tools" / "hadoop"


def _ensure_hadoop_home() -> None:
    if os.getenv("HADOOP_HOME"):
        return
    if (_HADOOP_HOME / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(_HADOOP_HOME.resolve())


def create_spark(app_name: str = "flood-ml") -> SparkSession:
    _ensure_hadoop_home()
    master = os.getenv("SPARK_MASTER", "local[*]")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g"))
        .config("spark.ui.enabled", os.getenv("SPARK_UI_ENABLED", "false"))
        # Bypass Windows NativeIO (hadoop.dll) issue — use algorithm v2 committer
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        # Disable Hadoop native libs on Windows to avoid UnsatisfiedLinkError
        .config("spark.hadoop.io.native.lib.available", "false")
    )
    return builder.getOrCreate()
