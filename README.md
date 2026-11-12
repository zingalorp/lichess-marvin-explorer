# Marvin Opening Explorer

A Chrome / Edge extension that adds a **Marvin** tab to the Lichess opening
explorer. It shows how human players at a chosen rating and time control are
likely to move, using the [Marvin](https://huggingface.co/holymolyyy/marvin-onnx)
model running locally on your PC.

![Marvin tab on the Lichess analysis board](docs/screenshot.png)

- **Policy**: how likely each legal move is to be played.
- **White / Draw / Black**: the predicted result after each move. **Σ** is the
  policy-weighted total.
- **Think time**: hover the clock next to a move to see its time distribution.
- Filters for rating (1200–2500) and time control, via the gear button.
- Works on the analysis board, finished games and studies.

Nothing is sent to any server: the model runs on your machine through Chrome's
native messaging.

## Install (Windows)

1. Download the latest `Marvin-Opening-Explorer-…-Windows.zip` from
   [Releases](https://github.com/zingalorp/lichess-marvin-explorer/releases/latest).
   It includes everything: no Python or other downloads needed.
2. Right-click the ZIP, choose **Extract All**, and keep the folder somewhere
   permanent (such as Documents).
3. Double-click **Setup.cmd** and follow the steps it shows. It opens your
   browser's extensions page, where you turn on **Developer mode**, click
   **Load unpacked** and choose the `extension` folder (its path is already
   copied for you).
4. Open an [analysis board](https://lichess.org/analysis), click the book button
   and choose the **Marvin** tab.

The toolbar icon chooses the model (79M or the faster 23M) and display options.
To remove it, double-click **Uninstall.cmd** and remove the extension in the
browser. `START-HERE.txt` in the ZIP has troubleshooting tips.

### From source (developers, GPU)

1. `python -m pip install -r host\requirements.txt` (Python 3.11+; use
   `onnxruntime-gpu` for an NVIDIA GPU).
2. Put `marvin_79m.onnx`, `marvin_23m.onnx` and their `.offset.npz` files from
   [holymolyyy/marvin-onnx](https://huggingface.co/holymolyyy/marvin-onnx) in a
   `weights` folder.
3. Run `.\install.ps1`, then load the `extension` folder as above.
   `.\install.ps1 -Uninstall` removes the host.

## Development

```powershell
python -m unittest discover -s tests -p "test_*.py"
node --test tests/
python tools/build_release.py   # builds the Windows ZIP into dist/
```
