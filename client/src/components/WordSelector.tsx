// src/components/WordSelector.tsx
import React, { useMemo } from 'react';
import Select, { MultiValue } from 'react-select'; // Import MultiValue
import { OptionType } from '../types'; // Import shared OptionType

import '../styles/WordSelector.scss';

interface WordSelectorProps {
    words: string[];
    selectedWords: string[]; // Changed from selectedWord: string
    onChange: (selectedOptions: MultiValue<OptionType>) => void; // Changed handler type
    disabled?: boolean;
    className?: string;
    id?: string;
}

const WordSelector: React.FC<WordSelectorProps> = ({
                                                       words,
                                                       selectedWords, // Use array
                                                       onChange,
                                                       disabled,
                                                       className,
                                                       id
                                                   }) => {

    const options = useMemo((): OptionType[] => {
        if (!words || words.length === 0) {
            return [];
        }
        return words.map(word => ({ value: word, label: word }));
    }, [words]);

    // Find the currently selected option objects based on the selectedWords array
    const selectedOptions = useMemo((): OptionType[] => {
        return options.filter(option => selectedWords.includes(option.value));
    }, [options, selectedWords]);

    const isDisabled = disabled || options.length === 0;

    const customSelectStyles = {
        control: (base: any) => ({
            ...base,
            minHeight: '36px',
            borderColor: 'var(--card-border-color, #dee2e6)',
            boxShadow: 'none',
            '&:hover': {
                borderColor: 'var(--bs-secondary-color, #6c757d)',
            },
        }),
        option: (base: any, { isSelected, isFocused }: any) => ({
            ...base,
            backgroundColor: isSelected ? 'var(--accent-color, #0d6efd)' : isFocused ? 'var(--bs-secondary-bg, #e9ecef)' : 'transparent',
            color: isSelected ? 'white' : 'var(--text-color, #212529)',
            '&:active': {
                backgroundColor: isSelected ? 'var(--accent-color, #0d6efd)' : 'var(--bs-light, #f8f9fa)',
            },
            fontSize: '0.9rem',
            padding: '0.5rem 0.75rem',
        }),
        input: (base: any) => ({
            ...base,
            color: 'var(--text-color, #212529)',
        }),
        // multiValue / singleValue styling might be needed
        multiValue: (base: any) => ({
            ...base,
            backgroundColor: 'var(--bs-secondary-bg, #e9ecef)',
            borderRadius: 'var(--bs-border-radius-sm)'
        }),
        multiValueLabel: (base : any) => ({
            ...base,
            color: 'var(--bs-secondary-color, #6c757d)',
            fontSize: '85%',
            paddingLeft: '0.5em',
            paddingRight: '0.3em'
        }),
        multiValueRemove: (base: any) => ({
            ...base,
            color: 'var(--bs-secondary-color, #6c757d)',
            '&:hover': {
                backgroundColor: 'var(--bs-danger-bg-subtle)',
                color: 'var(--bs-danger-text-emphasis)',
            },
        }),
        menu: (base: any) => ({
            ...base,
            zIndex: 3,
            backgroundColor: 'var(--card-bg, white)',
            border: `1px solid var(--card-border-color, #dee2e6)`,
            boxShadow: `var(--card-shadow, 0 2px 8px rgba(0, 0, 0, 0.06))`
        }),
        placeholder: (base: any) => ({
            ...base,
            color: 'var(--text-muted-color, #6c757d)'
        }),
    };

    return (
        <div className={`word-selector-group ${className || ''}`}>
            <Select<OptionType, true> // Specify isMulti via true generic
                inputId={id}
                isMulti // Enable multi-select
                options={options}
                value={selectedOptions}
                onChange={onChange}
                isDisabled={isDisabled}
                placeholder={isDisabled && words.length === 0 ? "-- Load data first --" : "-- Select or type words --"}
                isSearchable={true}
                closeMenuOnSelect={false} // Keep menu open for multi-select
                classNamePrefix="react-select"
                styles={customSelectStyles}
            />
        </div>
    );
};

export default WordSelector;