from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


VALUE_SHIFT = 1099511627776
DELTA_SHIFT = 2199023255552


@dataclass(frozen=True)
class PublicStatement:
    """Public statement for the proof."""

    expected_change: int
    tolerance: int


@dataclass(frozen=True)
class ProofBundle:
    """Generated proof and public signals."""

    proof: dict
    public_signals: list[str]


class ZkToolError(RuntimeError):
    """Raised when an external zk tool fails."""


def run_command(args: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess and raise on failure.

    :param args: Command and arguments.
    :param cwd: Working directory.
    :raises ZkToolError: If the command fails.
    """
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ZkToolError(
            f"Command failed: {' '.join(args)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


class Prover:
    """Groth16 prover for the net-change circuit."""

    def __init__(
        self,
        wasm_dir: Path,
        zkey_path: Path,
        work_dir: Path | None = None,
    ) -> None:
        """Initialize the prover.

        :param wasm_dir: Directory containing the compiled Circom WASM artifacts.
        :param zkey_path: Path to the final .zkey proving key.
        :param work_dir: Optional working directory for temporary files.
        """
        self.wasm_dir = wasm_dir
        self.zkey_path = zkey_path
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="zk_net_change_"))

        self.wasm_file = self.wasm_dir / "net_change.wasm"
        self.witness_js = self.wasm_dir / "generate_witness.js"

        if not self.wasm_file.exists():
            raise FileNotFoundError(f"Missing WASM file: {self.wasm_file}")
        if not self.witness_js.exists():
            raise FileNotFoundError(f"Missing witness generator: {self.witness_js}")
        if not self.zkey_path.exists():
            raise FileNotFoundError(f"Missing proving key: {self.zkey_path}")

    @staticmethod
    def encode_private_value(value: int) -> int:
        """Encode a signed private endpoint.

        :param value: Signed integer endpoint.
        :return: Shifted nonnegative value.
        """
        return value + VALUE_SHIFT

    @staticmethod
    def encode_expected_change(value: int) -> int:
        """Encode a signed expected net change.

        :param value: Signed integer expected net change.
        :return: Shifted nonnegative value.
        """
        return value + DELTA_SHIFT

    def prove(
        self,
        first_value: int,
        last_value: int,
        expected_change: int,
        tolerance: int,
    ) -> tuple[PublicStatement, ProofBundle]:
        """Generate a Groth16 proof.

        :param first_value: Private first value.
        :param last_value: Private last value.
        :param expected_change: Public expected net change.
        :param tolerance: Public tolerance.
        :return: Public statement and proof bundle.
        """
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        input_payload = {
            "first_shifted": str(self.encode_private_value(first_value)),
            "last_shifted": str(self.encode_private_value(last_value)),
            "expected_shifted": str(self.encode_expected_change(expected_change)),
            "tolerance": str(tolerance),
        }

        input_path = self.work_dir / "input.json"
        witness_path = self.work_dir / "witness.wtns"
        proof_path = self.work_dir / "proof.json"
        public_path = self.work_dir / "public.json"

        input_path.write_text(json.dumps(input_payload), encoding="utf-8")

        run_command(
            [
                "node",
                str(self.witness_js),
                str(self.wasm_file),
                str(input_path),
                str(witness_path),
            ]
        )

        run_command(
            [
                "snarkjs",
                "groth16",
                "prove",
                str(self.zkey_path),
                str(witness_path),
                str(proof_path),
                str(public_path),
            ]
        )

        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        public_signals = json.loads(public_path.read_text(encoding="utf-8"))

        statement = PublicStatement(
            expected_change=expected_change,
            tolerance=tolerance,
        )
        bundle = ProofBundle(
            proof=proof,
            public_signals=public_signals,
        )
        return statement, bundle

    def cleanup(self) -> None:
        """Remove the working directory."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)


class Verifier:
    """Groth16 verifier for the net-change circuit."""

    def __init__(self, verification_key_path: Path) -> None:
        """Initialize the verifier.

        :param verification_key_path: Path to verification_key.json.
        """
        self.verification_key_path = verification_key_path
        if not self.verification_key_path.exists():
            raise FileNotFoundError(
                f"Missing verification key: {self.verification_key_path}"
            )

    def verify(self, bundle: ProofBundle) -> bool:
        """Verify a Groth16 proof.

        :param bundle: Proof and public signals.
        :return: True if the proof verifies.
        """
        with tempfile.TemporaryDirectory(prefix="zk_verify_") as temp_dir:
            temp_path = Path(temp_dir)
            proof_path = temp_path / "proof.json"
            public_path = temp_path / "public.json"

            proof_path.write_text(json.dumps(bundle.proof), encoding="utf-8")
            public_path.write_text(
                json.dumps(bundle.public_signals), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    "snarkjs",
                    "groth16",
                    "verify",
                    str(self.verification_key_path),
                    str(public_path),
                    str(proof_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            return result.returncode == 0 and "OK!" in result.stdout


def demo() -> None:
    """Show proof generation and verification."""
    prover = Prover(
        wasm_dir=Path("build/net_change_js"),
        zkey_path=Path("build/net_change_final.zkey"),
    )
    verifier = Verifier(
        verification_key_path=Path("build/verification_key.json"),
    )

    statement, bundle = prover.prove(
        first_value=100,
        last_value=87,
        expected_change=-12,
        tolerance=2,
    )

    accepted = verifier.verify(bundle)

    print("Expected change:", statement.expected_change)
    print("Tolerance:", statement.tolerance)
    print("Public signals:", bundle.public_signals)
    print("Accepted:", accepted)

    prover.cleanup()


if __name__ == "__main__":
    demo()
