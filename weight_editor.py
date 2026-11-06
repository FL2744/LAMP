"""Guided raw-weight editor. Edits affect memory tensors only, never the base."""
import re


def read_text(prompt):
    try:
        return input(prompt).strip() or None
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def show_row(memory, matrix, slot):
    weight = getattr(memory, matrix)
    enabled = bool(memory.occupied[slot].item())
    print(f'\n{matrix.upper()} — shape {memory.slots} × {memory.width} — editing row {slot}')
    print(f'Memory {slot}: {"enabled" if enabled else "disabled"}; '
          f'matching threshold {memory.thresholds[slot].item():.9g}')
    values = weight[slot].detach().cpu().tolist()
    print('Column: value (columns start at 0)')
    for start in range(0, memory.width, 4):
        print('  ' + ' '.join(f'{i:02d}: {values[i]:.9g}'.ljust(19)
                             for i in range(start, min(start + 4, memory.width))).rstrip())
    print(f'Row length: {weight[slot].norm().item():.6g}')


def snapshot(memory):
    return {name: value.detach().clone() for name, value in memory.state_dict().items()}


def edit_memory_weights(e, memory_path):
    memory = e.model.memory
    history = []
    matrix = slot = None
    print('\nMEMORY WEIGHT EDITOR')
    print(f'Two matrices, each {memory.slots} rows × {memory.width} columns. Row = memory number.')
    print('Keys control which questions match. Deltas control the hidden-state change.')
    print('Columns are hidden-state features, not vocabulary token IDs.')
    print('Values are saved exactly at FP32 precision; keys are not automatically normalized.')
    print('A larger key can increase its match score. Disabled rows never activate.')
    print('Editing weights does not enable a row. Use action 5 to enable a new memory.')
    print('Every edit saves automatically. Undo is available while this editor is open.')
    print('Press Enter on an empty input to cancel the current action.\n')

    def commit(action, description):
        before = snapshot(memory)
        try:
            action()
            e.save_memory(memory_path)
        except Exception:
            memory.load_state_dict(before)
            raise
        history.append(before)
        print('Saved: ' + description)

    while True:
        try:
            if matrix is None:
                print('Choose a matrix: 1 Recognition keys   2 Activation deltas   0 Main menu')
                selected = read_text('Matrix number > ')
                if selected is None or selected == '0':
                    print('Returning to the main menu. Saved edits are kept.')
                    return
                if selected not in ('1', '2'):
                    print('Choose 1, 2, or 0.')
                    continue
                matrix = 'keys' if selected == '1' else 'deltas'
                selected_row = read_text(f'Row / memory number (0–{memory.slots - 1}) > ')
                if selected_row is None:
                    matrix = None
                    continue
                try:
                    slot = int(selected_row)
                    memory._slot(slot)
                except ValueError:
                    matrix = slot = None
                    print(f'Choose a row from 0 to {memory.slots - 1}.')
                    continue

            show_row(memory, matrix, slot)
            print('\n1 Change one weight       2 Replace this entire row')
            print('3 Multiply this row       4 Set matching threshold')
            print('5 Enable/disable memory   6 Undo last edit')
            print('7 Test a question         8 Choose another matrix/row')
            print('0 Return to main menu')
            action = read_text('Weight editor action > ')
            if action is None or action == '0':
                print('Returning to the main menu. Saved edits are kept.')
                return
            if action == '8':
                matrix = slot = None
                continue
            if action == '1':
                column_text = read_text(f'Column to change (0–{memory.width - 1}) > ')
                if column_text is None:
                    continue
                column = int(column_text)
                if not 0 <= column < memory.width:
                    raise ValueError(f'Column must be from 0 to {memory.width - 1}')
                old = getattr(memory, matrix)[slot, column].item()
                value_text = read_text(f'{matrix}[{slot}, {column}] is {old:.9g}. New value > ')
                if value_text is None:
                    continue
                row = getattr(memory, matrix)[slot].detach().clone()
                row[column] = float(value_text)
                commit(lambda: memory.edit_row(matrix, slot, row),
                       f'{matrix}[{slot}, {column}] = {row[column].item():.9g} (was {old:.9g}).')
            elif action == '2':
                text = read_text(f'Paste {memory.width} numbers, separated by spaces or commas > ')
                if text is None:
                    continue
                values = [float(piece) for piece in re.split(r'[\s,]+', text) if piece]
                commit(lambda: memory.edit_row(matrix, slot, values), f'replaced {matrix} row {slot}.')
            elif action == '3':
                text = read_text('Multiply every value in this row by (for example 0.5, 2, or 0) > ')
                if text is None:
                    continue
                factor = float(text)
                row = getattr(memory, matrix)[slot].detach().clone() * factor
                commit(lambda: memory.edit_row(matrix, slot, row),
                       f'multiplied {matrix} row {slot} by {factor:g}.')
            elif action == '4':
                text = read_text('New matching threshold (greater than 0 and less than 1) > ')
                if text is None:
                    continue
                threshold = float(text)
                commit(lambda: memory.set_slot_options(slot, threshold=threshold),
                       f'memory {slot} threshold = {threshold:g}.')
            elif action == '5':
                text = read_text('Type on to enable this memory, or off to disable it > ')
                if text is None:
                    continue
                if text.lower() not in ('on', 'off'):
                    raise ValueError('Type on or off')
                enabled = text.lower() == 'on'
                commit(lambda: memory.set_slot_options(slot, enabled=enabled),
                       f'memory {slot} {"enabled" if enabled else "disabled"}.')
            elif action == '6':
                if not history:
                    print('There are no edits to undo in this editor session.')
                    continue
                current = snapshot(memory)
                try:
                    memory.load_state_dict(history[-1])
                    e.save_memory(memory_path)
                except Exception:
                    memory.load_state_dict(current)
                    raise
                history.pop()
                print('Undid the last edit and saved the restored weights and settings.')
            elif action == '7':
                question = read_text('Question to test > ')
                if question is None:
                    continue
                before, after = e.ask(question, False), e.ask(question)
                print(f"Original answer: {before['answer']}\nWith memory: {after['answer']}")
                if after['memory_active']:
                    print(f"Matched memory {after['slot']}; score {after['similarity']:.6g}.")
                else:
                    score = after['similarity']
                    print('No memory matched.' + (f' Best score: {score:.6g}.' if score is not None else ''))
            else:
                print('Choose an action from 0 to 8.')
        except (ValueError, TypeError, RuntimeError, OverflowError, OSError) as error:
            print(f'Edit not saved: {error}')
