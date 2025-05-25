import React from 'react';
import { getVADDescription, VADDimension } from '../vadUtils';
import '../styles/VADTooltip.scss';

interface TooltipProps {
    active?: boolean;
    payload?: any[];
    label?: string | number;
    dimension?: VADDimension;
    showAll?: boolean;
}

const VADTooltip: React.FC<TooltipProps> = ({ active, payload, label, dimension, showAll = false }) => {
    if (active && payload && payload.length) {
        const dataPoint = payload[0].payload;

        return (
            <div className="vad-custom-tooltip">
                <p className="label">{`Year: ${label}`}</p>
                {showAll ? (
                    payload.map((entry, index) => {
                        const dim = entry.name as VADDimension;
                        const value = entry.value as number | null;
                        const description = getVADDescription(dim, value);
                        return (
                            <p key={index} className="desc" style={{ color: entry.color }}>
                                {`${dim}: ${value?.toFixed(3) ?? 'N/A'}`}
                                {description && <span className="qualitative">({description})</span>}
                            </p>
                        );
                    })
                ) : dimension ? (
                    payload.map((entry, index) => {
                        let content = null;
                        if (entry.name === dimension) {
                            const value = entry.value as number | null;
                            const description = getVADDescription(dimension, value);
                            content = (
                                <p key={`${index}-val`} className="desc" style={{ color: entry.color }}>
                                    {`${dimension}: ${value?.toFixed(3) ?? 'N/A'}`}
                                    {description && <span className="qualitative">({description})</span>}
                                </p>
                            );
                        }
                        else if (entry.dataKey === 'proportion') {
                            const value = entry.value as number | null;
                            content = (
                                <p key={`${index}-prop`} className="desc proportion" style={{ color: entry.color }}>
                                    {`Sense Prop.: ${value?.toFixed(3) ?? 'N/A'}`}
                                </p>
                            );
                        }
                        return content;
                    })
                ) : null}
            </div>
        );
    }
    return null;
};

export default VADTooltip;