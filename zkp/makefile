CIRCOM := $(HOME)/.cargo/bin/circom
SNARKJS := snarkjs

CIRCUIT := net_change
BUILD_DIR := build

CIRCOM_FILE := $(CIRCUIT).circom
R1CS := $(BUILD_DIR)/$(CIRCUIT).r1cs
SYM := $(BUILD_DIR)/$(CIRCUIT).sym
WASM_DIR := $(BUILD_DIR)/$(CIRCUIT)_js
WASM := $(WASM_DIR)/$(CIRCUIT).wasm
WITNESS_GEN := $(WASM_DIR)/generate_witness.js

PTAU_POWER := 12
PTAU_0 := $(BUILD_DIR)/pot$(PTAU_POWER)_0000.ptau
PTAU_1 := $(BUILD_DIR)/pot$(PTAU_POWER)_0001.ptau
PTAU_FINAL := $(BUILD_DIR)/pot$(PTAU_POWER)_final.ptau

ZKEY_0 := $(BUILD_DIR)/$(CIRCUIT)_0000.zkey
ZKEY_FINAL := $(BUILD_DIR)/$(CIRCUIT)_final.zkey
VK := $(BUILD_DIR)/verification_key.json

.PHONY: all compile ptau setup contribute vk clean

all: compile ptau setup contribute vk

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

compile: $(BUILD_DIR)
	$(CIRCOM) $(CIRCOM_FILE) --r1cs --wasm --sym -o $(BUILD_DIR)

$(PTAU_0): | $(BUILD_DIR)
	$(SNARKJS) powersoftau new bn128 $(PTAU_POWER) $(PTAU_0) -v

$(PTAU_1): $(PTAU_0)
	$(SNARKJS) powersoftau contribute $(PTAU_0) $(PTAU_1) \
		--name="First contribution" -v -e="$$(date +%s)-$$(hostname)"

$(PTAU_FINAL): $(PTAU_1)
	$(SNARKJS) powersoftau prepare phase2 $(PTAU_1) $(PTAU_FINAL) -v

ptau: $(PTAU_FINAL)

setup: compile ptau
	$(SNARKJS) groth16 setup $(R1CS) $(PTAU_FINAL) $(ZKEY_0)

contribute: setup
	$(SNARKJS) zkey contribute $(ZKEY_0) $(ZKEY_FINAL) \
		--name="Circuit contribution" -v -e="$$(date +%s)-$$(hostname)"

vk: contribute
	$(SNARKJS) zkey export verificationkey $(ZKEY_FINAL) $(VK)

clean:
	rm -rf $(BUILD_DIR)