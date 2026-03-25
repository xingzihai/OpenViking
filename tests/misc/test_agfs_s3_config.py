#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from openviking.agfs_manager import AGFSManager
from openviking_cli.utils.config.agfs_config import AGFSConfig, S3Config


def test_s3_nonempty_directory_marker_default_and_override(tmp_path):
    default_s3 = S3Config()
    assert default_s3.nonempty_directory_marker is False

    config = AGFSConfig(
        path=str(tmp_path),
        backend="s3",
        s3=S3Config(
            bucket="my-bucket",
            region="us-west-1",
            access_key="fake-access-key-for-testing",
            secret_key="fake-secret-key-for-testing-12345",
            endpoint="https://tos-cn-beijing.volces.com",
            nonempty_directory_marker=True,
        ),
    )

    assert config.s3.nonempty_directory_marker is True


def test_agfs_manager_passes_nonempty_directory_marker_to_s3fs(tmp_path):
    config = AGFSConfig(
        path=str(tmp_path),
        backend="s3",
        s3=S3Config(
            bucket="my-bucket",
            region="us-west-1",
            access_key="fake-access-key-for-testing",
            secret_key="fake-secret-key-for-testing-12345",
            endpoint="https://tos-cn-beijing.volces.com",
            nonempty_directory_marker=True,
        ),
    )

    manager = AGFSManager(config=config)
    agfs_config = manager._generate_config()

    assert agfs_config["plugins"]["s3fs"]["config"]["nonempty_directory_marker"] is True
