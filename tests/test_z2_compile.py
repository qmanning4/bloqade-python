from bloqade.atom_arrangement import Chain
import numpy as np


def test_z2_compile():
    # Define relevant parameters for the lattice geometry and pulse schedule
    n_atoms = 11
    lattice_spacing = 6.1 # (μm)
    min_time_step = 0.05 # (μs)

    # Choose a max Rabi amplitude of 15.8 MHz... pretty high.
    # This way, we minimize the protocol duration, 
    # but maintain the same pulse area, Ω*t.
    rabi_amplitude_values = [0.0, 15.8, 15.8, 0.0]

    # The lattice spacing and Rabi amplitudes give us a nearest-neighbor interaction strength:
        # V_{i}{i+1} = 105 MHz >> 15.8 MHz = Ω
    # Our interaction strength for next-nearest-neighbor is quite low comparitively:
        # V_{i}{i+2} = 1.64 MHz << 15.8 MHz = Ω
    # So far, this looks good for creating a Z2 topology

    # Next, define the detuning values
    rabi_detuning_values = [-16.33, -16.33, 16.33, 16.33]

    # We start at negative values to push the atom into ground state
    # We sweep to positive values to try and push the atom into Rydberg state
    # This gives us a static Rydberg blockade radius R_b = 8.32μm
    # Typically, we set lattice_spacing < R_b for a better blockade approximation
    # that no two atoms within lattice_spacing are both in the Rydberg state

    # Note the addition of a "sweep_time" variable
    # for performing sweeps of time values.
    durations = [0.8, "sweep_time", 0.8]

    # Create chain formation of our atoms and piecewise pulse shapes
    time_sweep_z2_prog = (
        Chain(n_atoms, lattice_spacing=lattice_spacing)
        .rydberg.rabi.amplitude.uniform.piecewise_linear(
            durations, rabi_amplitude_values
        )
        .detuning.uniform.piecewise_linear(durations, rabi_detuning_values)
    )

    # Allow "sweep_time" to assume values from 0.05 to 2.4 microseconds for a total of
    # 20 possible values.
    # Starting at exactly 0.0 isn't feasible so we use the `min_time_step` defined
    # previously.
    time_sweep_z2_job = time_sweep_z2_prog.batch_assign(
        sweep_time=np.linspace(min_time_step, 2.4, 20)
    )

    # Query on emulators and Aquila...
    bloqade_emu_target = time_sweep_z2_job.bloqade.python()
    braket_emu_target = time_sweep_z2_job.braket.local_emulator()
    quera_aquila_target = time_sweep_z2_job.parallelize(24).quera.aquila()
    braket_aquila_target = time_sweep_z2_job.parallelize(24).braket.aquila()

    targets = [
        bloqade_emu_target,
        braket_emu_target,
        quera_aquila_target,
        braket_aquila_target,
    ]

    # And, compile!
    for target in targets:
        target._compile(10)