import re
import os
import random


def parse_dl_file(file_path):
    """Parse the .dl file to extract arguments, attacks, and supports."""
    with open(file_path, 'r') as file:
        content = file.read()

    args = re.findall(r'arg\(([^,]+),\s*([^)]+)\)', content)
    attacks = re.findall(r'att\(([^,]+),\s*([^)]+)\)', content)
    supports = re.findall(r'sup\(([^,]+),\s*([^)]+)\)', content)

    args = [(a.strip(), v.strip()) for a, v in args]
    attacks = [(s.strip(), t.strip()) for s, t in attacks]
    supports = [(s.strip(), t.strip()) for s, t in supports]

    return args, attacks, supports


def identify_argument_types(args, attacks, supports):
    """Identify input, intermediate, and output arguments."""
    all_args = set(a[0] for a in args)
    targeted = set(t for _, t in attacks) | set(t for _, t in supports)
    sources = set([s for s, _ in attacks] + [s for s, _ in supports])

    output_args = targeted - sources
    intermediate_args = targeted & sources
    input_args = all_args - targeted

    return sorted(input_args), sorted(intermediate_args), sorted(output_args)


def generate_per_input_hidden_layers(input_args, hidden_size):
    """
    For EACH input, build its own block:
        input_k -> [h1_k ... hN_k] -> hidden_{input_k}
    Blocks are independent of each other.

    Returns: (hidden_nodes, new_input_nodes, layer_edges, input_mapping, blocks)
    """
    hidden_nodes = []
    new_input_nodes = []
    layer_edges = []
    input_mapping = {}
    blocks = {}

    for inp in input_args:
        new_name = f"hidden_{inp}"
        input_mapping[inp] = new_name
        new_input_nodes.append((new_name, 0.5))

        block_hidden = []
        for i in range(1, hidden_size + 1):
            h_name = f"h{i}_{inp}"
            hidden_nodes.append((h_name, 0.5))
            block_hidden.append(h_name)

        for h in block_hidden:
            layer_edges.append((inp, h))
        for h in block_hidden:
            layer_edges.append((h, new_name))

        blocks[inp] = {'hidden_nodes': block_hidden, 'new_input': new_name}

    return hidden_nodes, new_input_nodes, layer_edges, input_mapping, blocks


def generate_fully_connected_input_layer(input_args, hidden_size):
    """
    Single hidden layer SHARED by all inputs.

    With N inputs and `hidden_size` hidden nodes per input in the per-input
    version, this creates N * hidden_size hidden nodes (h1..hM) where:
      - EVERY input connects to EVERY hidden node
      - EVERY hidden node connects to EVERY hidden_{input} argument

    Returns: (hidden_nodes, new_input_nodes, layer_edges, input_mapping, info)
    """
    total_hidden = hidden_size * len(input_args)

    hidden_nodes = [(f"h{i}", 0.5) for i in range(1, total_hidden + 1)]
    hidden_names = [name for name, _ in hidden_nodes]

    input_mapping = {inp: f"hidden_{inp}" for inp in input_args}
    new_input_nodes = [(new_name, 0.5) for new_name in input_mapping.values()]

    layer_edges = []

    # every input -> every hidden node
    for inp in input_args:
        for h in hidden_names:
            layer_edges.append((inp, h))

    # every hidden node -> every new input argument
    for h in hidden_names:
        for new_name in input_mapping.values():
            layer_edges.append((h, new_name))

    info = {
        'inputs': list(input_args),
        'hidden_nodes': hidden_names,
        'new_inputs': list(input_mapping.values()),
    }

    return hidden_nodes, new_input_nodes, layer_edges, input_mapping, info


def generate_dl_file(args, input_args, intermediate_args, output_args,
                     attacks, supports,
                     hidden_nodes, new_input_nodes, layer_edges,
                     input_mapping, output_filename):
    """
    Write the new .dl file.

    Original edges whose source is an input are rewritten to start from
    hidden_{input}. All other original edges are kept. Edges inside the new
    hidden structure get a random attack/support type.
    """
    original_values = dict(args)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w') as f:
        for name in input_args + intermediate_args + output_args:
            f.write(f"arg({name},{original_values.get(name, 0.5)})\n")

        for name, value in hidden_nodes:
            f.write(f"arg({name},{value})\n")
        for name, value in new_input_nodes:
            f.write(f"arg({name},{value})\n")

        for source, target in attacks:
            f.write(f"att({input_mapping.get(source, source)},{target})\n")
        for source, target in supports:
            f.write(f"sup({input_mapping.get(source, source)},{target})\n")

        for source, target in layer_edges:
            if random.randint(0, 1) == 0:
                f.write(f"att({source},{target})\n")
            else:
                f.write(f"sup({source},{target})\n")


def visualize_blocks(blocks, hidden_size):
    print(f"\n{'='*60}")
    print(f"PER-INPUT HIDDEN BLOCKS (size: {hidden_size} per input)")
    print(f"{'='*60}\n")
    for inp, data in blocks.items():
        print(f"{inp} -> [{', '.join(data['hidden_nodes'])}] -> {data['new_input']}")
        print(f"    {hidden_size} edges (input->hidden) + "
              f"{hidden_size} edges (hidden->{data['new_input']})")
    print()


def visualize_fully_connected(info, hidden_size):
    inputs = info['inputs']
    hidden = info['hidden_nodes']
    new_inputs = info['new_inputs']

    print(f"\n{'='*60}")
    print(f"FULLY CONNECTED INPUT LAYER (size: {hidden_size} per input, "
          f"{len(hidden)} total shared)")
    print(f"{'='*60}\n")
    print(f"Input arguments ({len(inputs)}): {', '.join(inputs)}")
    print(f"Hidden nodes ({len(hidden)}): {', '.join(hidden)}")
    print(f"New input arguments ({len(new_inputs)}): {', '.join(new_inputs)}")
    print(f"  {len(inputs)} inputs x {len(hidden)} hidden = "
          f"{len(inputs) * len(hidden)} edges (input->hidden)")
    print(f"  {len(hidden)} hidden x {len(new_inputs)} new inputs = "
          f"{len(hidden) * len(new_inputs)} edges (hidden->new input)")
    print()


def print_stats(filename, input_args, intermediate_args, output_args,
                hidden_nodes, new_input_nodes, layer_edges):
    total_args = (len(input_args) + len(intermediate_args) + len(output_args)
                  + len(hidden_nodes) + len(new_input_nodes))
    print(f"✅ Generated {filename}")
    print(f"  Total arguments: {total_args}")
    print(f"  Hidden nodes: {len(hidden_nodes)}")
    print(f"  New input arguments: {len(new_input_nodes)}")
    print(f"  New layer edges: {len(layer_edges)}")


def main(seed=None):
    if seed is not None:
        random.seed(seed)

    file_path = 'envs/elevator_scenario.dl'  # Change this to your file path
    output_dir = 'envs/inputs'

    args, attacks, supports = parse_dl_file(file_path)
    input_args, intermediate_args, output_args = identify_argument_types(
        args, attacks, supports
    )

    print(f"Input arguments: {input_args}")
    print(f"Intermediate arguments: {intermediate_args}")
    print(f"Output arguments: {output_args}")

    hidden_sizes = [5, 10, 20, 50]

    for size in hidden_sizes:
        # ---------------- Per-input blocks ----------------
        print(f"\n{'='*60}")
        print(f"Generating per-input hidden layers with {size} nodes per input...")
        print(f"{'='*60}")

        hidden_nodes, new_input_nodes, layer_edges, input_mapping, blocks = \
            generate_per_input_hidden_layers(input_args, size)

        visualize_blocks(blocks, size)

        output_filename = os.path.join(output_dir, f"hidden_{size}.dl")
        generate_dl_file(
            args, input_args, intermediate_args, output_args,
            attacks, supports,
            hidden_nodes, new_input_nodes, layer_edges,
            input_mapping, output_filename
        )
        print_stats(output_filename, input_args, intermediate_args, output_args,
                    hidden_nodes, new_input_nodes, layer_edges)

        # ---------------- Fully connected (shared) ----------------
        print(f"\n{'='*60}")
        print(f"Generating FULLY CONNECTED layer ({size} per input x "
              f"{len(input_args)} inputs)...")
        print(f"{'='*60}")

        fc_hidden, fc_new_inputs, fc_edges, fc_mapping, fc_info = \
            generate_fully_connected_input_layer(input_args, size)

        visualize_fully_connected(fc_info, size)

        fc_filename = os.path.join(output_dir, f"hidden_{size}_fc.dl")
        generate_dl_file(
            args, input_args, intermediate_args, output_args,
            attacks, supports,
            fc_hidden, fc_new_inputs, fc_edges,
            fc_mapping, fc_filename
        )
        print_stats(fc_filename, input_args, intermediate_args, output_args,
                    fc_hidden, fc_new_inputs, fc_edges)


if __name__ == "__main__":
    main()