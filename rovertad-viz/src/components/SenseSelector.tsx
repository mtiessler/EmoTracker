import React from 'react';
import Form from 'react-bootstrap/Form';

import '../styles/SenseSelector.scss';

interface SenseSelectorProps {
    senses: string[];
    selectedSenseId: string;
    onChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
    disabled: boolean;
    className?: string;
    id?: string;
}

const SenseSelector: React.FC<SenseSelectorProps> = ({
                                                         senses,
                                                         selectedSenseId,
                                                         onChange,
                                                         disabled,
                                                         className,
                                                         id
                                                     }) => {
    return (
        <div className={`sense-selector-group ${className || ''}`}>
            <Form.Select
                id={id}
                aria-label="Select sense"
                value={selectedSenseId}
                onChange={onChange}
                disabled={disabled || senses.length === 0}
            >
                <option value="">-- Select a Sense --</option>
                {senses.map((senseId) => (
                    <option key={senseId} value={senseId}>
                        {senseId.length > 45 ? senseId.substring(0, 42) + '...' : senseId}
                    </option>
                ))}
            </Form.Select>
            <Form.Text className="sense-helper-text d-block mt-1">
                Selects sense for 4D plot coloring and info display.
            </Form.Text>
        </div>
    );
};

export default SenseSelector;