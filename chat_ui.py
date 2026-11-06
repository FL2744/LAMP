"""Plain-language terminal interface for the toy memory experiment."""


MENU_ACTIONS = (
    ('1', '/ask', 'Ask a question'),
    ('2', '/write', 'Add/change a memory'),
    ('3', '/compare', 'Compare answers'),
    ('4', '/base', 'Ask original model'),
    ('5', '/examples', 'Example questions'),
    ('6', '/answers', 'Available answers'),
    ('7', '/slots', 'Saved memories'),
    ('8', '/clear', 'Remove memories'),
    ('9', '/help', 'Help'),
    ('10', '/vocab', 'Complete vocabulary'),
    ('11', '/weights', 'Edit memory weights'),
    ('0', '/quit', 'Quit'),
)
MENU_COMMANDS = {number: command for number, command, _ in MENU_ACTIONS}


def show_menu():
    print('\nACTIONS — type a number and press Enter, or type your question directly.')
    for i in range(0, len(MENU_ACTIONS), 2):
        pair = MENU_ACTIONS[i:i + 2]
        left = f'{pair[0][0]:>2}  {pair[0][2]}'
        if len(pair) == 2:
            right = f'{pair[1][0]:>2}  {pair[1][2]}'
            print(f'  {left:<31} {right}')
        else:
            print(f'  {left}')
    print()


def answer_label(answer):
    if answer == 'losangeles':
        return 'Los Angeles'
    return answer.capitalize()


def show_help():
    print('''
HOW TO USE MEMORY LAB

Every main prompt shows a numbered menu with all actions. Enter a number to
choose an action; the interface will ask for anything else it needs.

Ask a question by typing it normally. You do not need a command or quotation marks.
  What is the capital of Texas?
  Are all cats black?

You can also use slash commands instead of the menu:
  /ask       Ask a question with saved memories (menu 1).
  /write     Add or change a memory. I will ask for the question and then the answer.
  /compare   See the original answer and the answer with your saved memories.
  /base      Ask a question using only the original model, ignoring saved memories.
  /examples  Show the questions this small toy understands.
  /answers   Show the answers you can choose when writing a memory.
  /vocab     Show every vocabulary token and its ID (menu 10).
  /weights   Inspect and edit the actual memory matrices (menu 11).
  /slots     List the numbers of your saved memories.
  /clear     Choose a memory to remove, or remove them all.
  /help      Show these instructions again.
  /quit      Exit. Your saved memories will still be here next time.

TRY IT
  1. Choose menu 2 (Add/change a memory), or type /write.
  2. When asked for the question, type: What is the capital of Texas?
  3. When asked for the new answer, type: Sacramento
  4. Ask: What is the capital of Texas?
  5. Ask: Are all cats black?  That unrelated answer should stay the same.

Writes and removals save automatically. Press Enter on an empty guided prompt
to cancel that action and return to the menu. At the main prompt, numbers pick
actions. When removing a memory, the guided prompt asks for a memory number.

OPTIONAL SHORTCUTS
  /write What is the capital of Texas? => Sacramento
    The question goes before =>; your chosen answer goes after it.
  /compare What is the capital of Texas?
  /base What is the capital of Texas?
  /clear 0     Remove the memory numbered 0 (use /slots to find its number).
  /clear all   Remove every saved memory. The original model stays available.
''')


def prompt_text(label):
    try:
        text = input(label).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        text = ''
    if not text:
        print('Canceled. Nothing was changed.')
        return None
    return text


def show_answers(answers):
    print('\nAnswers available in this toy:')
    print('  ' + ', '.join(answer_label(a) for a in sorted(answers)))
    print('Capitalization does not matter. You can type Los Angeles with a space.')


def show_vocabulary(tokenizer):
    vocabulary = tokenizer.vocabulary
    print(f'\nCOMPLETE VOCABULARY — {len(vocabulary)} tokens')
    print('Each entry shows the token ID followed by its exact stored spelling.\n')
    for start in range(0, len(vocabulary), 3):
        entries = [f'{i:>2}  {vocabulary[i]:<17}'
                   for i in range(start, min(start + 3, len(vocabulary)))]
        print('  ' + '  '.join(entries).rstrip())
    print('\nSpecial tokens:')
    explanations = {
        '<pad>': 'Padding used to give questions in a batch the same length.',
        '<unk>': 'Represents a word outside this vocabulary.',
        '<bos>': 'Marks the beginning of a question.',
        '<answer>': 'Marks the position where the model predicts its answer.',
    }
    for token, explanation in explanations.items():
        if token in vocabulary:
            print(f'  {token:<10} {explanation}')
    print('\nQuestions are lowercased before tokenization. Type words, not token IDs.')
    print('For the answers you can choose when writing a memory, use menu 6.')


def show_examples(facts, templates):
    print('\nTry any of these questions. Just type the question, without a command:')
    for kind, subject, _ in facts:
        if kind in ('capital', 'largest'):
            subject = subject.title()
        question = templates[kind][0].format(s=subject)
        print('  ' + question[0].upper() + question[1:])
    print('\nDifferent wording may work, but this toy only covers these topics.')
    print('Use /write to change an answer, or /compare to see a memory\'s effect.')


def show_slots(e):
    memory = e.model.memory
    slots = memory.occupied.nonzero().flatten().tolist()
    print(f'\n{len(slots)} of {memory.slots} memory spaces are in use.')
    for slot in slots:
        print(f'  Memory {slot} is saved. To remove it, type /clear {slot}')
    if not slots:
        print('Type /write to add your first memory.')
    else:
        print('These numbers identify saved memories; they are not answer choices.')
        print('Use /compare with a question to see which memory it activates.')
    return slots


def show_answer(e, question, use_memory=True):
    result = e.ask(question, use_memory=use_memory)
    print(f"Answer: {answer_label(result['answer'])}")
    if not use_memory:
        print('Saved memories were ignored for this question.')
    elif result['memory_active']:
        print(f"Used memory {result['slot']}. Type /compare to see the original answer.")
    else:
        print('No saved memory matched; this is the original model\'s answer.')


def run_chat(e, threshold, memory_path, facts, templates):
    if memory_path.exists():
        e.load_memory(memory_path)
    answers = {answer for _, _, answer in facts}
    count = int(e.model.memory.occupied.sum().item())
    print('\nMEMORY LAB')
    print('Ask a question, then try changing its answer with a saved memory.')
    print('This small toy covers capital cities, largest cities, and a few questions about animals.')
    noun = 'memory' if count == 1 else 'memories'
    print(f'{count} saved {noun} loaded. Changes save automatically.')
    print('\nStart by typing a question: What is the capital of Texas?')
    print('Choose 2 (Add/change a memory) to change an answer. I will guide you through it.')
    print('All actions are listed below and return after each question or action.')

    while True:
        show_menu()
        try:
            line = input('You > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye. Saved memories are kept for next time.')
            break
        if not line:
            continue
        if line in MENU_COMMANDS:
            line = MENU_COMMANDS[line]
        elif line.isdecimal():
            print('Choose a menu number from 0 to 11, or type a question.')
            continue
        if line.lower() in ('menu', '/menu'):
            continue
        if line.lower() in ('help', 'quit', 'exit'):
            line = '/' + ('quit' if line.lower() == 'exit' else line.lower())
        if not line.startswith('/'):
            try:
                show_answer(e, line)
            except (ValueError, RuntimeError) as error:
                print(f'I could not answer that: {error}')
                print('Type /examples for questions this toy understands.')
            continue

        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ''
        try:
            if command in ('/quit', '/exit'):
                print('Goodbye. Saved memories are kept for next time.')
                break
            elif command == '/help':
                show_help()
            elif command == '/examples':
                show_examples(facts, templates)
            elif command == '/answers':
                show_answers(answers)
            elif command in ('/vocab', '/vocabulary'):
                show_vocabulary(e.tokenizer)
            elif command == '/weights':
                from weight_editor import edit_memory_weights
                edit_memory_weights(e, memory_path)
            elif command in ('/write', '/remember'):
                question, separator, answer = argument.partition('=>')
                question = question.strip()
                if not question:
                    print('\nLet\'s write a memory. Press Enter on an empty response to cancel.')
                    question = prompt_text('Which question should this memory answer? ')
                if question is None:
                    continue
                if not separator or not answer.strip():
                    show_answers(answers)
                    answer = prompt_text('What answer should the model give for that question? ')
                if answer is None:
                    continue
                canonical = answer.strip().lower()
                if canonical == 'los angeles':
                    canonical = 'losangeles'
                if canonical not in answers:
                    print(f'{answer.strip()!r} is not an available answer in this toy. Nothing was saved.')
                    show_answers(answers)
                    print('Type /write to try again.')
                    continue
                info = e.write(question, canonical, threshold=threshold)
                e.save_memory(memory_path)
                print(f"Saved memory {info['slot']}.")
                print(f'  Question: {question}')
                print(f'  Chosen answer: {answer_label(canonical)}')
                print('Now ask that question normally, or type /compare to check the change.')
                print(f"To remove this memory, type /clear {info['slot']}")
            elif command in ('/ask', '/base', '/compare'):
                question = argument or prompt_text('Which question would you like to ask? ')
                if question is None:
                    continue
                if command in ('/ask', '/base'):
                    show_answer(e, question, use_memory=command == '/ask')
                else:
                    before = e.ask(question, use_memory=False)
                    after = e.ask(question)
                    print(f'Question: {question}')
                    print(f"  Original model:     {answer_label(before['answer'])}")
                    print(f"  With saved memories: {answer_label(after['answer'])}")
                    if after['memory_active']:
                        print(f"Memory {after['slot']} was used.")
                    else:
                        print('No saved memory matched this question.')
            elif command in ('/slots', '/memories'):
                show_slots(e)
            elif command == '/clear':
                slots = show_slots(e)
                if not slots:
                    continue
                selection = argument or prompt_text(
                    'Type a memory number to remove it, or all to remove every memory (Enter cancels): ')
                if selection is None:
                    continue
                if selection.lower() == 'all':
                    e.model.memory.clear()
                    message = 'Removed all saved memories.'
                else:
                    try:
                        slot = int(selection)
                    except ValueError:
                        print('Nothing was removed. Type /clear 0 for memory 0, or /clear all for every memory.')
                        continue
                    if slot not in slots:
                        print(f'There is no saved memory numbered {slot}. Nothing was removed.')
                        continue
                    e.model.memory.clear(slot)
                    message = f'Removed memory {slot}.'
                e.save_memory(memory_path)
                print(message + ' The original model is still available.')
            else:
                print(f'I do not recognize the command {parts[0]!r}.')
                print('Type /help to see the commands, or type a question without a slash.')
        except (ValueError, RuntimeError) as error:
            print(f'I could not complete that action: {error}')
            print('Type /help for instructions or /answers to see the supported answers.')
