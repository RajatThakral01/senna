# Detailed Implementation Plan

This document breaks down each phase of the project into specific, actionable steps, based on the `AGENT_CONTEXT.md` file.

## Phase 1: Implement Input Handling

1.  **Create Directory**: Create the `input/` directory.
2.  **Create File**: Create the `input/input_handler.py` file.
3.  **Implement Source Detection**: Add the `detect_source_type` function to `input_handler.py`.
4.  **Implement Handlers**: Add the handler functions for each source type (`youtube`, `youtube_live`, `google_drive`, `local_file`, `direct_url`) to `input_handler.py`. This will involve using `yt-dlp`, `gdown`, `requests`, and `shutil`.
5.  **Implement Main Handler**: Create the main `handle_input` function in `input_handler.py` that orchestrates the source detection and calls the appropriate handler.
6.  **Update Dependencies**: Add `gdown` and `requests` to the `requirements.txt` file.
7.  **Integrate with `main.py`**: Modify `main.py` to use `handle_input` to process the video input, replacing the direct call to the old `downloader.py`.

## Phase 2: Implement Campaign Parser

1.  **Create Directories**: Create the `campaign/` and `campaign/templates/` directories.
2.  **Create Initializer**: Create an empty `campaign/__init__.py` file.
3.  **Build Parser**: Create `campaign/campaign_parser.py` and implement the `parse_campaign` function, which calls the MiniMax API to convert natural language to a JSON config.
4.  **Create Templates**: Create the three JSON template files (`podcast_clip.json`, `tiktok_reaction.json`, `motivational_reel.json`) inside the `campaign/templates/` directory.
5.  **Implement Merging Logic**: Add the `load_template` and `merge_config` utility functions to `main.py` or a new utility file to handle the merging of templates and campaign configurations.

## Phase 3: Implement Dynamic Pipeline Logic

1.  **Extend `improvements.py`**: Add the new `apply_logo` and `mix_music` functions to the `improvements.py` file.
2.  **Update `main.py` CLI**: Modify `main.py` to accept the new `--campaign` and `--template` command-line arguments using `argparse`.
3.  **Implement Dynamic Execution**: In `main.py`, add conditional logic within the clip processing loop to call `apply_logo`, `mix_music`, and other functions based on the feature flags in the parsed campaign configuration.

## Phase 4: Set up Configuration and Asset Management

1.  **Create `config.yaml`**: Create the `config.yaml` file and populate it with all the hardcoded paths and settings from the project.
2.  **Create Asset Directories**: Create the `assets/`, `assets/logos/`, and `assets/music/` directories.
3.  **Implement Config Loading**: In `main.py`, add logic to load and use the settings from `config.yaml` using the `PyYAML` library.
4.  **Update Dependencies**: Add `pyyaml` and `python-dotenv` to `requirements.txt`.
5.  **Implement Asset Scanning**: Add a function to scan the `assets/` subdirectories to make them available to the campaign parser.

## Phase 5: Build the Gradio UI

1.  **Create `ui.py`**: Create the `ui.py` file.
2.  **Implement UI Layout**: Build the Gradio interface with all the specified components (text boxes, buttons, gallery, etc.).
3.  **Connect to Pipeline**: Wire the UI components to the `run_pipeline` function in `main.py`.
4.  **Add Progress and Logging**: Implement progress updates and logging to the UI to provide feedback to the user during processing.
5.  **Update Dependencies**: Add `gradio` to `requirements.txt`.
6.  **Launch Logic**: Add the `if __name__ == "__main__":` block to launch the Gradio app.

## Phase 6: Perform Integration Testing and Finalize

1.  **End-to-End Tests**: Run at least three complete tests using the Gradio UI, each with a different video source and campaign description.
2.  **Verify Outputs**: Check that the final video clips are generated correctly and that the `report.json` is accurate.
3.  **Cleanup Verification**: Ensure that all intermediate files are properly deleted after the pipeline completes.
4.  **Refactor and Review**: Review the new code for clarity, and refactor the old modules to be part of the new `pipeline/` directory structure as specified in the architecture diagram.
