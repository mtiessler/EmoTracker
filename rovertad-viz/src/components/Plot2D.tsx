import React, { useMemo } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    TooltipProps
} from 'recharts';
import Icon from '@mdi/react';
import { mdiAlertCircleOutline } from '@mdi/js';
import { CombinedDataPoint, LoadedData, SenseInfo } from '../types';
import { getVADDescription, VADDimension } from '../vadUtils';

import '../styles/Plot2D.scss';

interface Plot2DProps {
    selectedWords: string[];
    allWordsData: LoadedData | null;
    yLabel: VADDimension;
    selectedSenseData?: SenseInfo | null;
}

interface PayloadEntry {
    color: string;
    dataKey: string;
    name: string;
    value: number | null;
    payload: CombinedDataPoint;
    stroke: string;
    fill: string;
}

interface CustomTooltipProps extends TooltipProps<number | null, string> {
    active?: boolean;
    payload?: PayloadEntry[];
    label?: string | number;
    dimension?: VADDimension;
}

const lineColors = ['#0d6efd', '#6f42c1', '#198754', '#ffc107', '#dc3545', '#fd7e14', '#20c997', '#6610f2'];

const Plot2D: React.FC<Plot2DProps> = ({ selectedWords, allWordsData, yLabel, selectedSenseData }) => {

    const { chartData, yDomain, linesToPlot, hasValidData, timeDataRef, showProportionLine } = useMemo(() => {
        const combinedChartData: { [time: number]: CombinedDataPoint } = {};
        const allDimValues: number[] = [];
        const lines: { word: string; dataKey: string; color: string }[] = [];
        let timeRef: number[] | null = null;
        let validDataFound = false;
        const dimKey = yLabel.charAt(0).toLowerCase() as 'v' | 'a' | 'd';
        let propData: (number | null)[] | null = null;
        let showProp = false;

        if (!allWordsData || selectedWords.length === 0 || !['v', 'a', 'd'].includes(dimKey)) {
            return { chartData: [], yDomain: [0, 1] as [number | string, number | string], linesToPlot: [], hasValidData: false, timeDataRef: null, showProportionLine: false };
        }

        const singleWord = selectedWords.length === 1 ? selectedWords[0] : null;
        if (singleWord && selectedSenseData?.y_fitting) {
            const wordData = allWordsData[singleWord];
            if (wordData?.temporal_vad?.x && selectedSenseData.y_fitting.length === wordData.temporal_vad.x.length) {
                propData = selectedSenseData.y_fitting.map(p => (typeof p === 'number' && !isNaN(p) ? p : null));
                showProp = true;
            }
        }

        selectedWords.forEach((word, wordIndex) => {
            const wordData = allWordsData[word];
            const vadData = wordData?.temporal_vad;

            if (!vadData?.x || !vadData[dimKey]) {
                console.warn(`Missing VAD data (${yLabel}) for word: ${word}`);
                return;
            }

            const x = vadData.x;
            const valueData = vadData[dimKey];

            if (!timeRef) {
                timeRef = x;
            } else if (JSON.stringify(timeRef) !== JSON.stringify(x)) {
                console.warn(`Time data mismatch for word: ${word}. Skipping.`);
                return;
            }

            validDataFound = true;
            const color = lineColors[wordIndex % lineColors.length];
            const dataKey = `${word}_value`;
            lines.push({ word: word, dataKey: dataKey, color: color });

            x.forEach((time, index) => {
                if (!combinedChartData[time]) {
                    combinedChartData[time] = { time };
                }
                const dimVal = typeof valueData[index] === 'number' && !isNaN(valueData[index]) ? valueData[index] : null;
                combinedChartData[time][dataKey] = dimVal;
                if (dimVal !== null) allDimValues.push(dimVal);
                if (showProp && propData && word === singleWord) {
                    combinedChartData[time]['proportion'] = propData[index];
                }
            });
        });

        const finalChartData = Object.values(combinedChartData).sort((p1, p2) => p1.time - p2.time);
        let finalYDomain: [number | string, number | string] = [0, 1];

        if (allDimValues.length > 0) {
            const minValue = Math.min(...allDimValues);
            const maxValue = Math.max(...allDimValues);
            const range = maxValue - minValue;
            const padding = range === 0 ? 0.1 : range * 0.05;
            const calculatedMin = minValue - padding;
            const calculatedMax = maxValue + padding;

            if (calculatedMin < calculatedMax) {
                if (yLabel === 'Valence' || yLabel === 'Arousal' || yLabel === 'Dominance') {
                    if (calculatedMin < 0 || calculatedMax > 1) {
                        finalYDomain = [calculatedMin, calculatedMax];
                    } else {
                        finalYDomain = [Math.max(0, calculatedMin), Math.min(1, calculatedMax)];
                    }
                } else {
                    finalYDomain = [calculatedMin, calculatedMax];
                }
            } else if (allDimValues.length === 1) {
                finalYDomain = [allDimValues[0] - padding, allDimValues[0] + padding];
            }
        }

        return { chartData: finalChartData, yDomain: finalYDomain, linesToPlot: lines, hasValidData: validDataFound, timeDataRef: timeRef, showProportionLine: showProp };

    }, [selectedWords, allWordsData, yLabel, selectedSenseData]);


    if (selectedWords.length === 0) {
        return (
            <div className="plot-placeholder info">
                <h4>Select Word(s)</h4>
                <p>Select one or more words to view the {yLabel} plot.</p>
            </div>
        );
    }

    if (!hasValidData) {
        return (
            <div className="plot-placeholder error">
                <Icon path={mdiAlertCircleOutline} size={1.6} className="placeholder-icon" />
                <h4>Data Error</h4>
                <p>No valid {yLabel} data found for the selected word(s).</p>
            </div>
        );
    }

    const MultiWordTooltip: React.FC<CustomTooltipProps> = ({ active, payload, label, dimension }) => {
        if (active && payload && payload.length && dimension) {
            return (
                <div className="vad-custom-tooltip multi-line">
                    <p className="label">{`Year: ${label}`}</p>
                    {payload.map((entry: PayloadEntry, index: number) => {
                        let content = null;
                        if (entry.dataKey === 'proportion') {
                            const value = entry.value;
                            content = (
                                <p key={`${index}-prop`} className="desc proportion" style={{ color: entry.color }}>
                                    {`Sense Prop.: ${value?.toFixed(3) ?? 'N/A'}`}
                                </p>
                            );
                        } else {
                            const word = entry.name;
                            const value = entry.value;
                            const description = getVADDescription(dimension, value);
                            content = (
                                <p key={index} className="desc" style={{ color: entry.color }}>
                                    {`${word}: ${value?.toFixed(3) ?? 'N/A'}`}
                                    {description && <span className="qualitative">({description})</span>}
                                </p>
                            );
                        }
                        return content;
                    })}
                </div>
            );
        }
        return null;
    };

    return (
        <div className="plot2d-wrapper">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 35, left: 10, bottom: 35 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--grid-color, #e0e0e0)" />
                    <XAxis
                        dataKey="time"
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        allowDuplicatedCategory={false}
                        tickCount={Math.min(10, timeDataRef?.length ?? 10)}
                        label={{ value: "Year", position: "insideBottom", dy: 15 }}
                        height={50}
                        scale="time"
                        interval="preserveStartEnd"
                        tickLine={false}
                    />
                    <YAxis
                        yAxisId="left"
                        domain={yDomain}
                        label={{ value: yLabel, angle: -90, position: 'insideLeft', dx: -5 }}
                        width={65}
                        tickFormatter={(tick) => typeof tick === 'number' ? tick.toFixed(2) : tick}
                        tickLine={false}
                    />
                    {showProportionLine && (
                        <YAxis
                            yAxisId="right"
                            orientation="right"
                            domain={[0, 1]}
                            label={{ value: 'Sense Prop.', angle: 90, position: 'insideRight', dx: 5 }}
                            width={65}
                            tickFormatter={(tick) => typeof tick === 'number' ? tick.toFixed(2) : tick}
                            tickLine={false}
                        />
                    )}
                    <Tooltip
                        offset={15}
                        cursor={{ stroke: 'var(--text-muted-color)', strokeDasharray: '3 3' }}
                        content={<MultiWordTooltip dimension={yLabel} />}
                    />
                    <Legend verticalAlign="top" height={36} />
                    {linesToPlot.map(lineInfo => (
                        <Line
                            yAxisId="left"
                            key={lineInfo.dataKey}
                            type="monotone"
                            dataKey={lineInfo.dataKey}
                            name={lineInfo.word}
                            stroke={lineInfo.color}
                            strokeWidth={1.5}
                            activeDot={{ r: 4, strokeWidth: 0, fill: lineInfo.color }}
                            dot={false}
                            connectNulls={false}
                        />
                    ))}
                    {showProportionLine && (
                        <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="proportion"
                            name="Sense Prop."
                            stroke="var(--bs-secondary-color, #6c757d)"
                            strokeWidth={1}
                            strokeDasharray="5 5"
                            activeDot={{ r: 4, strokeWidth: 0, fill: 'var(--bs-secondary-color, #6c757d)' }}
                            dot={false}
                            connectNulls={false}
                        />
                    )}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

export default Plot2D;