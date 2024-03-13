import argparse
import copy
import os
import json
import tqdm
import random


def build_sgd_api_dict_v0(apis):
    def _get_slots_info(slot_dict, slots, slot_type):
        slots_info = []
        for _slot in slots:
            slot_info = slot_dict[_slot]
            slots_info.append(slot_info)

        # return {slot_type + ' parameters': '; '.join(slots_info)}
        return {slot_type + ' parameters': slots_info}

    api_dict, api_param_dict = {}, {}
    for service in apis:
        service_name = service['service_name']
        service_description = service['description']
        slot_dict = {}
        api_dict[service_name] = {}
        api_param_dict[service_name] = {}
        for slot in service['slots']:
            s_name, s_description, s_is_categorical, s_values = \
                slot['name'], slot['description'], slot['is_categorical'], slot['possible_values']
            slot_info = {
                "name": slot['name'],
                "description": slot['description'],
                "is_categorical": slot['is_categorical'],
                "possible_values": process_possible_values(slot['possible_values'])
            }
            slot_dict[s_name] = slot_info

        for intent in service['intents']:
            func_name, func_description = intent['name'], intent['description']
            is_transactional = intent['is_transactional']
            required_slots, optional_slots, result_slots = intent['required_slots'], \
                                                           intent['optional_slots'], intent['result_slots']

            tmp_obj = {'API name': func_name, 'description': func_description, 'is_transactional': is_transactional}
            func_info = json.dumps(
                tmp_obj | _get_slots_info(slot_dict, required_slots, 'Required') | \
                _get_slots_info(slot_dict, optional_slots, 'Optional')
            )
            api_dict[service_name][func_name] = func_info.replace('true', 'True').replace('false', 'False').replace('{',
                                                                                                                    '(').replace(
                '}', ')')

            ### Save required and optional parameters
            api_param_dict[service_name][func_name] = {'required': [], 'optional': []}
            for req_param in required_slots:
                api_param_dict[service_name][func_name]['required'].append(req_param)
            for opt_param in optional_slots:
                api_param_dict[service_name][func_name]['optional'].append(opt_param)

    return api_dict, api_param_dict


def process_possible_values(values):
    out = []
    for v in values:
        if v.isdigit():
            out.append(int(v))
        elif v == 'True':
            out.append(True)
        elif v == 'False':
            out.append(False)
        else:
            out.append(v)
    return out

def build_sgd_api_dict(apis, the_only_service=None, the_only_intent=None, shuffle_idx=True, use_index=True, json_instruct=False):
    def _get_slots_info(slot_dict, slots, slot_type):
        slots_info = []
        for _slot in slots:
            print(slot_dict.keys())
            slot_info = slot_dict[_slot]
            slots_info.append(slot_info)

        # return {slot_type + ' parameters': '; '.join(slots_info)}
        return {slot_type + ' parameters': slots_info}

    def _build_d3st_slot(idx, desc, is_categorical, categorical_values, json_instruct):
        if json_instruct:
            d3st_slot = {'s'+str(idx): {'description': desc}}
        else:
            d3st_slot = 's'+str(idx) + ': ' + desc
        if not is_categorical:
            return d3st_slot, None
        else:
            categorical_value_idx_dict = {}
            d3st_slot_list, d3st_ctg_value_dict = [], {}
            random.shuffle(categorical_values)
            for _ic, categorical_value in enumerate(categorical_values):
                d3st_slot_list.append('s' + str(idx) + '.' + str(_ic) + ': ' + categorical_value)
                categorical_value_idx_dict[categorical_value] = 's' + str(idx) + '.' + str(_ic)
                d3st_ctg_value_dict['s' + str(idx) + '.' + str(_ic)] = categorical_value

            if json_instruct:
                d3st_slot['s'+str(idx)]['categorical_values'] = d3st_ctg_value_dict
            else:
                d3st_slot += ' {' + ', '.join(d3st_slot_list) + '}'
            return d3st_slot, categorical_value_idx_dict

    def _build_d3st_intention(idx, intent_description, required_slots, optional_slots, json_instruct):
        if json_instruct:
            d3st_intent = {'i'+str(idx): intent_description}
        else:
            d3st_intent = 'i'+str(idx) + ': ' + intent_description
        # if len(required_slots) > 0:
        #     d3st_intent += ', ' + ' '.join(required_slots) + ' required'
        # if len(optional_slots) > 0:
        #     d3st_intent += ', ' + ' '.join(optional_slots) + ' optional'
        return d3st_intent

    def _get_slots_idx(slots, slot_idx_dict):
        slots_idx = []
        for slot in slots:
            slots_idx.append(slot_idx_dict[slot])
        return slots_idx

    api_dict, api_param_dict, func_param2idx_dict, categorical_value2idx_dict = {}, {}, {}, {}
    idx2func_param_dict, idx2categorical_value_dict = {}, {}

    for _service in apis:
        if the_only_service is None:
            service = _service
        else:
            service = the_only_service
        service_name = service['service_name']
        service_description = service['description']
        slot_dict = {}
        if json_instruct:
            api_dict[service_name] = {'slot_values': {}, 'intents': {}, 'description': ''}
        else:
            api_dict[service_name] = []
        api_param_dict[service_name] = {}
        func_param2idx_dict[service_name], categorical_value2idx_dict[service_name] = {'intent': {}, 'slot': {}}, {}
        idx2func_param_dict[service_name], idx2categorical_value_dict[service_name] = {}, {}
        slots = service['slots']
        if shuffle_idx:
            random.shuffle(slots)

        intents = service['intents']
        slots_interested = []
        if the_only_intent is not None:
            for _it, intent in enumerate(intents):
                func_name, func_description = intent['name'], intent['description']
                if func_name != the_only_intent:
                    continue
                required_slots, optional_slots = intent['required_slots'], intent['optional_slots']
                for _s in required_slots:
                    slots_interested.append(_s)
                for _s, _ in optional_slots.items():
                    slots_interested.append(_s)
                break

        for _is, slot in enumerate(slots):
            s_name, s_description, s_is_categorical, s_values = \
                slot['name'], slot['description'], slot['is_categorical'], slot['possible_values']
            if the_only_intent is not None and s_name not in slots_interested:
                continue
            if use_index:
                d3st_form, categorical_value_idx_dict = _build_d3st_slot(_is, s_description, s_is_categorical, s_values, json_instruct)
                if categorical_value_idx_dict is not None:
                    categorical_value2idx_dict[service_name][s_name] = categorical_value_idx_dict
                    for cv, cv_idx in categorical_value_idx_dict.items():
                        idx2categorical_value_dict[service_name][cv_idx] = cv
            else:
                d3st_form = s_name + ': ' + s_description
                if s_is_categorical:
                    d3st_form += ' {Values: ' + ', '.join(s_values)+'}'

            slot_info = {
                "name": slot['name'],
                "description": slot['description'],
                "is_categorical": slot['is_categorical'],
                "possible_values": process_possible_values(slot['possible_values']),
                "d3st_form": d3st_form
            }
            func_param2idx_dict[service_name]['slot'][s_name] = 's'+str(_is)
            idx2func_param_dict[service_name]['s'+str(_is)] = s_name
            # slot_info = json.dumps({
            #     "name": slot['name'],
            #     "description": slot['description'],
            #     "is_categorical": slot['is_categorical'],
            #     "possible_values": process_possible_values(slot['possible_values'])
            # })
            slot_dict[s_name] = slot_info
            if json_instruct:
                api_dict[service_name]['slot_values'] = api_dict[service_name]['slot_values'] | d3st_form
            else:
                api_dict[service_name].append(d3st_form)

        if shuffle_idx:
            random.shuffle(intents)
        for _it, intent in enumerate(intents):
            func_name, func_description = intent['name'], intent['description']
            if the_only_intent is not None and func_name != the_only_intent:
                continue
            is_transactional = intent['is_transactional']
            required_slots, optional_slots, result_slots = intent['required_slots'], \
                intent['optional_slots'], intent['result_slots']

            tmp_obj = {'API name': func_name, 'description': func_description, 'is_transactional': is_transactional}
            # func_info = json.dumps(
            #     tmp_obj | _get_slots_info(slot_dict, required_slots, 'Required') | \
            #     _get_slots_info(slot_dict, optional_slots, 'Optional')
            # )

            # api_dict[service_name][func_name] = func_info.replace('true', 'True').replace('false', 'False').replace('{', '(').replace('}', ')')

            ### Save required and optional parameters
            api_param_dict[service_name][func_name] = {'required': [], 'optional': []}
            for req_param in required_slots:
                api_param_dict[service_name][func_name]['required'].append(req_param)
            for opt_param in optional_slots:
                api_param_dict[service_name][func_name]['optional'].append(opt_param)
            func_param2idx_dict[service_name]['intent'][intent['name']] = 'i' + str(_it)
            if json_instruct:
                api_dict[service_name]['intents'] = api_dict[service_name]['intents'] | _build_d3st_intention(
                    _it,
                    intent['description'],
                    _get_slots_idx(required_slots, func_param2idx_dict[service_name]['slot']),
                    _get_slots_idx(optional_slots, func_param2idx_dict[service_name]['slot']),
                    json_instruct=True
                )
            else:
                api_dict[service_name].append(_build_d3st_intention(
                    _it,
                    intent['description'],
                    _get_slots_idx(required_slots, func_param2idx_dict[service_name]['slot']),
                    _get_slots_idx(optional_slots, func_param2idx_dict[service_name]['slot']),
                    json_instruct=False
                ))

                api_dict[service_name] = intent['description'] + '. Parameters: ' + '; '.join(api_dict[service_name][:-1])
            break

        if the_only_service is not None:
            break

    return api_dict, api_param_dict, func_param2idx_dict, categorical_value2idx_dict, idx2func_param_dict, idx2categorical_value_dict


def build_api_dict(api_list):
    api_dict = {}
    for api in api_list:
        api_dict[api['service_name']] = api
    return api_dict


def load_sgd(path, use_index, use_long_api, json_instruct,
             prompt=None, all_intents_instructions=False):

    sgd_processed_data, full_sgd_processed_data = {}, {}
    all_api_calls = {'train': {}, 'dev': {}, 'test': {}}
    all_api_params = {'train': {}, 'dev': {}, 'test': {}}
    for split in ['train', 'dev', 'test']:
        fdir = os.path.join(path, split)
        all_apis = json.load(open(os.path.join(fdir, "schema.json"), 'r'))
        all_apis_dict = build_api_dict(all_apis)
        api_dict_in_split, api_param_dict_in_split = build_sgd_api_dict_v0(all_apis)
        all_api_params[split] = api_param_dict_in_split
        dps, full_dps = [], []
        for root, subdirs, files in os.walk(fdir):
            for file in files:
                if file == 'schema.json':
                    continue
                fpath = os.path.join(fdir, file)
                _data = json.load(open(fpath, 'r'))
                for dialogue in _data:
                    dialogue_id = dialogue['dialogue_id']
                    turns = dialogue['turns']
                    all_utterance = []
                    for _it, turn in enumerate(turns):
                        frames, utterance, speaker = turn['frames'], turn['utterance'], turn['speaker']
                        all_utterance.append(speaker + ': ' + utterance)
                        assert len(frames) <= 2, frames
                        for _if, frame in enumerate(frames[-1:]):

                            if 'state' in frame.keys():
                                assert _if == 0, (turn)
                                service_name = frame['service']
                                intent = frame['state']['active_intent']
                                if intent == 'NONE':
                                    continue

                                chat = ' '.join(all_utterance)
                                if prompt is not None:
                                    chat = '"' + chat + '"' + (' ' + prompt)

                                api_dict, api_param_dict, func_param2idx_dict, categorical_value2idx_dict, \
                                    idx2func_param_dict, idx2categorical_value_dict = build_sgd_api_dict(
                                    all_apis,
                                    the_only_service=all_apis_dict[service_name],
                                    the_only_intent=intent,
                                    shuffle_idx=True,
                                    use_index=use_index,
                                    json_instruct=json_instruct
                                )

                                params, param_keys, params_dict, params_dict_w_desc = [], [], {}, {}
                                for _param_key, _param_values in frame['state']['slot_values'].items():
                                    param_keys.append(_param_key)
                                    _param_value = _param_values[0]
                                    if _param_key not in func_param2idx_dict[service_name]['slot']:
                                        continue
                                    if use_index:
                                        _param_key_idx = func_param2idx_dict[service_name]['slot'][_param_key]
                                        if _param_key in categorical_value2idx_dict[
                                            service_name].keys() and _param_value != 'dontcare':  # categorical param
                                            params.append(_param_key_idx + ': ' +
                                                          categorical_value2idx_dict[service_name][_param_key][
                                                              _param_value])



                                            params_dict[_param_key_idx] = \
                                                categorical_value2idx_dict[service_name][_param_key][_param_value]
                                        else:
                                            params.append(_param_key_idx + ': ' + _param_value)

                                            params_dict[_param_key_idx] = _param_value
                                    else:
                                        params_dict[_param_key] = _param_values[0]

                                if len(list(params_dict.keys())) == 0:
                                    continue
                                if use_long_api:
                                    api = api_dict_in_split[service_name][intent]
                                else:
                                    api = api_dict[service_name]

                                idx2intent_dict = {}
                                if not json_instruct:
                                    api_documentation = 'API documentation: ' + api.replace('\\"', '"').replace('\"', '"')
                                else:
                                    api['description'] = 'API documentation: ' + api['description']
                                    api_documentation = json.dumps(api)

                                input, instruction = chat, api_documentation

                                output_dict, detailed_output_dict = {}, {}

                                for _param_key in params_dict.keys():
                                    output_dict[_param_key] = params_dict[_param_key]

                                api_call = json.dumps(output_dict) # + ' [intent] ' + func_idx

                                dps.append({
                                    'uid': dialogue_id + ':' + str(_it),
                                    'input': input,
                                    'instruction': instruction,
                                    'output': api_call,
                                    'dialogue_id': dialogue_id,
                                    'turn': _it,
                                    'service': service_name,
                                    'function': intent,
                                    'parameters': param_keys,
                                })

                                full_dps.append({
                                    'uid': dialogue_id + ':' + str(_it),
                                    'input': input,
                                    'instruction': instruction,
                                    'output': api_call,
                                    'dialogue_id': dialogue_id,
                                    'turn': _it,
                                    'service': service_name,
                                    'function': intent,
                                    'parameters': param_keys,
                                    'idx2func_param_dict': idx2func_param_dict[service_name],
                                    'idx2categorical_value_dict': idx2categorical_value_dict[service_name],
                                    'idx2intent_dict': idx2intent_dict
                                })

                                # if split == 'train':
                                if service_name not in all_api_calls[split]:
                                    all_api_calls[split][service_name] = {}

                                if intent not in all_api_calls[split][service_name]:
                                    all_api_calls[split][service_name][intent] = [api_call]
                                else:
                                    all_api_calls[split][service_name][intent].append(api_call)

                        # all_utterance.append(speaker + ': ' + utterance)
        for x in dps:
            a = x['dialogue_id']
        sgd_processed_data[split] = dps
        full_sgd_processed_data[split] = full_dps
        print(split, len(dps))

    return sgd_processed_data, full_sgd_processed_data

def get_all_params_funcs(processed_data):
    all_params, all_funcs = [], []
    for instance in processed_data:
        params, func = instance['parameters'], instance['function']
        if func not in all_funcs:
            all_funcs.append(func)
        for param in params:
            if param not in all_params:
                all_params.append(param)
    return all_params, all_funcs


def main(args):
    data_dir = os.path.join('.', 'raw_data', 'dstc8-schema-guided-dialogue')
    if args.dataset_name == 'sgd':

        prompt = None
        if args.add_prompt is not None:
            if args.add_prompt == "v1":
                prompt = "Now, generate the API call to fulfill the USER's request following the API documentation. Today is 2019-03-01."
            else:
                raise NotImplementedError

        sgd_processed_data, full_sgd_processed_data = load_sgd(
            data_dir,
            use_index=args.use_index,
            use_long_api=args.use_long_api,
            prompt=prompt,
            json_instruct=args.json_instruction,
        )

        out_dataset_name = args.dataset_name+'_d3st'
        if args.add_prompt:
            out_dataset_name += '_prompt'
        if args.json_instruction:
            out_dataset_name += '_jsonInstruct'
        if not args.use_index:
            out_dataset_name += "_NoIndex"
        if args.use_long_api:
            out_dataset_name += '_LongAPI'

        for split in ['train', 'dev', 'test']:
            out_data_dir = os.path.join('.', out_dataset_name)
            if not os.path.exists(out_data_dir):
                os.makedirs(out_data_dir)
            with open(os.path.join(out_data_dir, split+'.json'), 'w') as f:
                json.dump(sgd_processed_data[split], f, indent=2)

        for split in ['train', 'dev', 'test']:
            out_data_dir = os.path.join('.', 'full_data', out_dataset_name)
            if not os.path.exists(out_data_dir):
                os.makedirs(out_data_dir)
            with open(os.path.join(out_data_dir, split+'.json'), 'w') as f:
                json.dump(full_sgd_processed_data[split], f, indent=2)

    elif args.dataset_name == 'sgd_x':
        data_dir = os.path.join(data_dir, 'sgd_x', 'data')
        prompt = None
        if args.add_prompt is not None:
            if args.add_prompt == "v1":
                prompt = "Now, generate the API call to fulfill the USER's request following the API documentation. Today is 2019-03-01."
            else:
                raise NotImplementedError

        out_dataset_name = args.dataset_name + '_d3st'
        if args.add_prompt:
            out_dataset_name += '_prompt'
        if args.json_instruction:
            out_dataset_name += '_jsonInstruct'

        if args.use_long_api:
            out_dataset_name += '_LongAPI'
        if not args.use_index:
            out_dataset_name += "_NoIndex"

        for vnum in range(1,6):
            sgd_processed_data, full_sgd_processed_data = load_sgd(
                os.path.join(data_dir, 'v'+str(vnum)),
                use_index=args.use_index,
                use_long_api=args.use_long_api,
                prompt=prompt,
                json_instruct=args.json_instruction,
            )

            for split in ['train', 'dev', 'test']:
                out_data_dir = os.path.join('.', out_dataset_name, 'v'+str(vnum))
                if not os.path.exists(out_data_dir):
                    os.makedirs(out_data_dir)
                with open(os.path.join(out_data_dir, split+'.json'), 'w') as f:
                    json.dump(sgd_processed_data[split], f, indent=2)

            for split in ['train', 'dev', 'test']:
                out_data_dir = os.path.join('.', 'full_data', out_dataset_name, 'v'+str(vnum))
                if not os.path.exists(out_data_dir):
                    os.makedirs(out_data_dir)
                with open(os.path.join(out_data_dir, split + '.json'), 'w') as f:
                    json.dump(full_sgd_processed_data[split], f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=['sgd', 'sgd_x'])
    parser.add_argument("--add_prompt", type=str, default=None,
                        choices=["v1", "v2", "v3"])
    parser.add_argument("--use_index", action="store_true")
    parser.add_argument("--use_long_api", action="store_true")
    parser.add_argument("--json_instruction", action="store_true")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)