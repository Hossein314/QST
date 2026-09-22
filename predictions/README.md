Save masks from the app here with Ctrl+S, then compare them against
`ground_truth/`:

    python tools/diff_masks.py predictions/ ground_truth/ diffs/ --images samples/
    python tools/compare_viewer.py predictions/ ground_truth/ --images samples/
