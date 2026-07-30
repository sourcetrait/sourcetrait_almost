# consts.rs

## const DPO_MODEL_NAME

## const BASE_MODEL_NAME

The base exists for loader generality and provenance runs only. It is not a
second target: changing the checkpoint is itself an era change.

## const MODEL_AUTHOR

## const MODELS_HOME_RELATIVE

The models home sits under the XDG DATA home rather than the cache home,
because a checkpoint is not regenerable. Snapshots take the opposite call for
the opposite reason.

## const SUITE_CONFIG_RELATIVE

## const STOP_TOKENS

Resolved by STRING at runtime rather than pinned by id, so the tokenizer's own
map wins over any config-side drift. Two are needed rather than one because the
shipped generation config carries only the end-of-text token while a chat turn
ends on `<|im_end|>`, and a generation has to stop on either.

## const EOS_TOKEN

What the chat template closes its FINAL assistant turn with. An interior turn
closes on `<|im_end|>` instead, and that positional split is the whole reason
the renderer reports spans rather than measuring prefixes.

## const TOKEN_EXTRA_ID_0

The added-token ids the engine leans on. They are identical to era one's Olmo 3
map, which is what lets the typed-channel design carry across eras at all.

The BASE checkpoint does not share them: its 100266-100275 range holds
`extra_id_1..10` with no tool markers, because the Instruct SFT stage is what
renamed four reserved ids into the tool markers. So the tokenizer lock rejects
the base BY DESIGN, and `<functions>` is one token here and several there.

## const TOKEN_ENDOFTEXT

## const TOKEN_IM_START

## const TOKEN_IM_END

## const TOKEN_FUNCTIONS_OPEN

## const TOKEN_FUNCTIONS_CLOSE

## const TOKEN_FUNCTION_CALLS_OPEN

## const TOKEN_FUNCTION_CALLS_CLOSE

## const TOKEN_EXTRA_ID_1

## const TOKEN_EXTRA_ID_6

## const TOKEN_ENDOFPROMPT

The insufficiency tag, and the one id here whose `special` flag changes how it
must be handled: a skip-special decode strips it from text entirely, so
detection is by token id at the engine or plugin layer and never by scanning
the decoded string.

## const TOKEN_PAD
