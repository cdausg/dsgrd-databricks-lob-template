"""
Post-generation hook for cookiecutter.
Removes .azure/pipelines/ if the project uses GitHub Actions instead of Azure DevOps.
"""

import os
import shutil

CICD_PLATFORM = "{{cookiecutter.cicd_platform}}"

project_dir = os.getcwd()
azure_pipelines_dir = os.path.join(project_dir, ".azure")

if CICD_PLATFORM == "github_actions":
    shutil.rmtree(azure_pipelines_dir, ignore_errors=True)
