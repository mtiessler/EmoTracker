import React from 'react';
import Button from 'react-bootstrap/Button';
import { VizType } from '../types';

import '../styles/VizControl.scss'; // Corrected path

interface VizControlProps {
    selectedViz: VizType;
    onChange: (type: VizType) => void;
    disabled?: boolean;
    className?: string;
}

const VIZ_OPTIONS: { label: string; type: VizType }[] = [
    { label: '2D V', type: '2D-V' },
    { label: '2D A', type: '2D-A' },
    { label: '2D D', type: '2D-D' },
    { label: '2D VAD', type: '2D-VAD' },
    { label: '3D', type: '3D' },
    { label: '4D', type: '4D' },
    // { label: 'Heatmap', type: 'Spectrogram' }, // Removed
];

const VizControl: React.FC<VizControlProps> = ({ selectedViz, onChange, disabled, className }) => {
    return (
        <div className={`viz-control-group ${className || ''}`}>
            <div
                className="d-flex flex-wrap viz-button-container"
                role="group"
                aria-label="Visualization Type Selector"
            >
                {VIZ_OPTIONS.map((option) => (
                    <Button
                        key={option.type}
                        variant={selectedViz === option.type ? 'primary' : 'outline-secondary'}
                        onClick={() => onChange(option.type)}
                        className="viz-button"
                        disabled={disabled}
                        size="sm"
                    >
                        {option.label}
                    </Button>
                ))}
            </div>
        </div>
    );
};

export default VizControl;