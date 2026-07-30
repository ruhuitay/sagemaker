# Implementation Plan: Multi-Cloud Minimal

## Overview

4 coding tasks plus a README update, each completable in 5-10 minutes. Total ~150 lines of new/modified code. No tests in this iteration.

## Tasks

- [ ] 1. Create config loader and example config file
  - Create `src/app/config.py` with a `load_config()` function (~25 lines)
  - Reads `config.json` from project root (2 levels up from the file)
  - Falls back to env vars (`AWS_ENDPOINT_URL`, `AWS_API_KEY`, `ALICLOUD_ENDPOINT_URL`, `ALICLOUD_API_TOKEN`)
  - Returns dict with shape `{"aws": {"url": "", "key": ""}, "alicloud": {"url": "", "key": ""}}`
  - Create `config.json.example` at project root with placeholder values
  - Add `config.json` to `.gitignore` (it contains secrets)
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 2. Modify DigitCanvas for provider selection and dynamic dispatch
  - Remove the global `TRITON_URL` constant from `src/app/main.py`
  - Import and call `load_config()` in `__init__`
  - Add `tk.StringVar` and radio buttons (AWS / Alicloud) after the button frame
  - Add `_default_provider()` helper that picks first provider with a non-empty URL
  - Modify `predict()` to read `self.provider_var.get()`, select URL and build headers with if/else
  - AWS uses `{"x-api-key": key}`, Alicloud uses `{"Authorization": key}`
  - Add early return with error message if selected provider has no URL configured
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_

- [ ] 3. Create Alicloud PAI-EAS deploy script
  - Create `scripts/deploy_alicloud.py` (~80 lines)
  - Use argparse with env var fallbacks for: `--model-path`, `--bucket`, `--oss-key`, `--region`, `--service-name`, `--instance-type`
  - Implement `upload_to_oss()` using `oss2` SDK to upload model repository files
  - Implement `create_eas_service()` using `alibabacloud_pai_eas20210701` SDK
  - Print endpoint URL and token on success, error message and `sys.exit(1)` on failure
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 4. Update README with multi-cloud usage instructions
  - Add a "Multi-Cloud Inference" section to README.md
  - Document `config.json` setup (copy from `config.json.example`)
  - Document env var fallback option
  - Document how to run the deploy script for Alicloud
  - List required pip packages for Alicloud: `oss2`, `alibabacloud-pai-eas20210701`, `alibabacloud-tea-openapi`
  - _Requirements: 1.1, 4.2_

- [ ] 5. Checkpoint - Verify everything works
  - Ensure `config.json.example` exists and is valid JSON
  - Ensure `config.json` is in `.gitignore`
  - Ensure DigitCanvas launches without errors when no config is present (falls back gracefully)
  - Ensure deploy script prints usage help with `--help`

## Notes

- No tests in this iteration - can be added as a follow-up
- No abstract base classes or provider factories - just if/else
- Both providers use the identical Triton V2 JSON body already built in existing code
- The only difference between providers is the URL and the auth header name
