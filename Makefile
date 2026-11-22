.PHONY: all clean test lint logo synth train blender eval backend plots smoke

PYTHON ?= python3
HEAD ?= gaussian
BACKBONE ?= resnet18
EPOCHS ?= 40
SEED ?= 42
MODE ?= B
VAL_DIR ?= data/blender/room1_medium_nominal

all: test

logo:
	$(PYTHON) scripts/run_stage.py --stage logo

synth:
	@echo ">>> Generating synthetic dataset splits (Train, Val, Calib)..."
	PYTHONPATH=. $(PYTHON) src/synth/generate.py --split train --num_images 10000 --workers 4 --out_dir data/synth/train
	PYTHONPATH=. $(PYTHON) src/synth/generate.py --split val   --num_images 1000  --workers 4 --out_dir data/synth/val
	PYTHONPATH=. $(PYTHON) src/synth/generate.py --split calib --num_images 1000  --workers 4 --out_dir data/synth/calib
	@echo ">>> Generating all 3 validation scenes/rooms..."
	PYTHONPATH=. $(PYTHON) src/blender/render.py --room room1_small_office --density medium --condition nominal --frames 300 --out_dir data/blender/room1_medium_nominal
	PYTHONPATH=. $(PYTHON) src/blender/render.py --room room2_lab_narrow   --density medium --condition nominal --frames 300 --out_dir data/blender/room2_lab_narrow
	PYTHONPATH=. $(PYTHON) src/blender/render.py --room room3_square_hall  --density medium --condition nominal --frames 300 --out_dir data/blender/room3_square_hall

train:
	$(PYTHON) scripts/train.py --head $(HEAD) --backbone $(BACKBONE) --epochs $(EPOCHS) --seed $(SEED)

train-all:
	@echo ">>> Training ALL 18 model combinations (6 heads x 3 backbones)..."
	for bb in resnet50; do \
		for head in gaussian gaussian_beta student_t student_t_fixed cauchy mse_fixed; do \
			echo "============================================================"; \
			echo ">>> Training Backbone: $$bb | Head: $$head (Epochs: $(EPOCHS), Seed: $(SEED))"; \
			echo "============================================================"; \
			$(PYTHON) scripts/train.py --head $$head --backbone $$bb --epochs $(EPOCHS) --seed $(SEED); \
		done \
	done

train-all-heads:
	@echo ">>> Training all 6 heads on Backbone: $(BACKBONE)..."
	for head in gaussian gaussian_beta student_t student_t_fixed cauchy mse_fixed; do \
		echo "============================================================"; \
		echo ">>> Training Backbone: $(BACKBONE) | Head: $$head"; \
		echo "============================================================"; \
		$(PYTHON) scripts/train.py --head $$head --backbone $(BACKBONE) --epochs $(EPOCHS) --seed $(SEED); \
	done

train-matrix-smoke:
	@echo ">>> Training all heads on ResNet-9, ResNet-18, ResNet-50 (smoke)..."
	for head in gaussian gaussian_beta student_t student_t_fixed cauchy mse_fixed; do \
		for bb in resnet9 resnet18; do \
			echo ">>> Training $$bb with $$head..."; \
			$(PYTHON) scripts/train.py --head $$head --backbone $$bb --epochs 6 --seed 42; \
		done \
	done

blender:
	$(PYTHON) scripts/run_stage.py --stage blender

eval:
	$(PYTHON) scripts/run_stage.py --stage eval

eval-all:
	@echo ">>> Evaluating ALL 18 model combinations on validation set..."
	for bb in resnet9 resnet18 resnet50; do \
		for head in gaussian gaussian_beta student_t student_t_fixed cauchy mse_fixed; do \
			ckpt="checkpoints/$${bb}_$${head}_seed$(SEED)/model_final.pt"; \
			out_json="checkpoints/$${bb}_$${head}_seed$(SEED)/eval_results.json"; \
			if [ -f "$$ckpt" ]; then \
				echo "============================================================"; \
				echo ">>> Evaluating Backbone: $$bb | Head: $$head"; \
				echo "============================================================"; \
				PYTHONPATH=. $(PYTHON) src/eval/evaluate.py --checkpoint "$$ckpt" --data_dir $(VAL_DIR) --backbone $$bb --head $$head --out "$$out_json"; \
			else \
				echo "Skipping missing checkpoint: $$ckpt"; \
			fi \
		done \
	done

eval-all-heads:
	@echo ">>> Evaluating all 6 heads on Backbone: $(BACKBONE)..."
	for head in gaussian gaussian_beta student_t student_t_fixed cauchy mse_fixed; do \
		ckpt="checkpoints/$(BACKBONE)_$${head}_seed$(SEED)/model_final.pt"; \
		out_json="checkpoints/$${BACKBONE}_$${head}_seed$(SEED)/eval_results.json"; \
		if [ -f "$$ckpt" ]; then \
			echo "============================================================"; \
			echo ">>> Evaluating Backbone: $(BACKBONE) | Head: $$head"; \
			echo "============================================================"; \
			PYTHONPATH=. $(PYTHON) src/eval/evaluate.py --checkpoint "$$ckpt" --data_dir $(VAL_DIR) --backbone $(BACKBONE) --head $$head --out "$$out_json"; \
		else \
			echo "Skipping missing checkpoint: $$ckpt"; \
		fi \
	done

backend:
	$(PYTHON) scripts/run_stage.py --stage backend

backend-ground-truth:
	@echo ">>> Running Backend Factor Graph Localization on ALL 18 model combinations (Mode $(MODE))..."
	for bb in resnet9 resnet18; do \
		for head in gaussian gaussian_beta student_t student_t_fixed cauchy mse_fixed; do \
			echo "============================================================"; \
			echo ">>> Backend Localization: Backbone: $$bb | Head: $$head | Mode: $(MODE)"; \
			echo "============================================================"; \
			PYTHONPATH=. $(PYTHON) src/backend/slam.py --data_dir $(VAL_DIR) --mode $(MODE) --head $$head --backbone $$bb; \
		done \
	done

backend-all:
	@echo ">>> Running Backend Factor Graph Localization on ALL 18 model combinations (Mode $(MODE))..."
	for bb in resnet9 resnet18 resnet50; do \
		for head in gaussian gaussian_beta student_t student_t_fixed cauchy mse_fixed; do \
			ckpt="checkpoints/$${bb}_$${head}_seed$(SEED)/model_final.pt"; \
			if [ -f "$$ckpt" ]; then \
				echo "============================================================"; \
				echo ">>> Backend Localization: Backbone: $$bb | Head: $$head | Mode: $(MODE)"; \
				echo "============================================================"; \
				PYTHONPATH=. $(PYTHON) src/backend/slam.py --data_dir $(VAL_DIR) --mode $(MODE) --head $$head --backbone $$bb --checkpoint "$$ckpt"; \
			else \
				echo "Skipping missing checkpoint: $$ckpt"; \
			fi \
		done \
	done

backend-all-heads:
	@echo ">>> Running Backend Factor Graph Localization on all 6 heads for Backbone: $(BACKBONE) (Mode $(MODE))..."
	for head in gaussian gaussian_beta student_t student_t_fixed cauchy mse_fixed; do \
		ckpt="checkpoints/$(BACKBONE)_$${head}_seed$(SEED)/model_final.pt"; \
		if [ -f "$$ckpt" ]; then \
			echo "============================================================"; \
			echo ">>> Backend Localization: Backbone: $(BACKBONE) | Head: $$head | Mode: $(MODE)"; \
			echo "============================================================"; \
			PYTHONPATH=. $(PYTHON) src/backend/slam.py --data_dir $(VAL_DIR) --mode $(MODE) --head $$head --backbone $(BACKBONE) --checkpoint "$$ckpt"; \
		else \
			echo "Skipping missing checkpoint: $$ckpt"; \
		fi \
	done

experiments:
	PYTHONPATH=. $(PYTHON) scripts/run_experiments.py

experiments-smoke:
	PYTHONPATH=. $(PYTHON) scripts/run_experiments.py --smoke

benchmark:
	PYTHONPATH=. $(PYTHON) src/eval/benchmark_and_plot.py

video:
	PYTHONPATH=. $(PYTHON) src/plots/make_video.py --backbone $(BACKBONE) --head $(HEAD) --mode $(MODE)

video-all:
	PYTHONPATH=. $(PYTHON) src/plots/make_video.py --all

plots:
	PYTHONPATH=. $(PYTHON) src/plots/make_paper_figures.py

paper-video:
	PYTHONPATH=. $(PYTHON) src/plots/make_video.py --backbone $(BACKBONE) --head $(HEAD) --mode $(MODE)

video-all:
	PYTHONPATH=. $(PYTHON) src/plots/make_video.py --all

plots:
	PYTHONPATH=. $(PYTHON) src/plots/make_paper_figures.py

smoke:
	$(PYTHON) scripts/run_smoke.py

test:
	pytest tests/ -v
