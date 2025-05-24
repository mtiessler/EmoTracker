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

import '../styles/Plot2DVAD.scss';

interface Plot2DVADProps {
    selectedWords: string[];
    allWordsData: LoadedData | null;
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
}

interface MemoizedPlotData {
    chartData: CombinedDataPoint[];
    yDomain: [number | string, number | string];
    linesToPlot: { word: string; dim: VADDimension; color: string }[];
    hasValidData: boolean;
    timeDataRef: number[] | null;
    showProportionLine: boolean;
}

const lineColors = ['#0d6efd', '#dc3545', '#198754', '#ffc107', '#6f42c1', '#fd7e14', '#20c997', '#6610f2'];

const Plot2DVAD: React.FC<Plot2DVADProps> = ({ selectedWords, allWordsData, selectedSenseData }) => {

    const { chartData, yDomain, linesToPlot, hasValidData, timeDataRef, showProportionLine }: MemoizedPlotData = useMemo(() => {
        const combinedChartData: { [time: number]: CombinedDataPoint } = {};
        const allValues: number[] = [];
        const lines: { word: string; dim: VADDimension; color: string }[] = [];
        let timeRef: number[] | null = null;
        let validDataFound = false;
        let propData: (number | null)[] | null = null;
        let showProp = false;

        if (!allWordsData || selectedWords.length === 0) {
            return { chartData: [], yDomain: [0, 1] as [number | string, number | string], linesToPlot: [], hasValidData: false, timeDataRef: null, showProportionLine: false };
        }

        const singleWord = selectedWords.length === 1 ? selectedWords[0] : null;

        if (singleWord && selectedSenseData?.y_fitting) {
            const wordDataCheck = allWordsData ? allWordsData[singleWord] : null;
            if (wordDataCheck?.temporal_vad?.x && selectedSenseData.y_fitting.length === wordDataCheck.temporal_vad.x.length) {
                propData = selectedSenseData.y_fitting.map(p => (typeof p === 'number' && !isNaN(p) ? p : null));
                if (propData.some(p => p !== null)) {
                    showProp = true;
                }
            }
        }

        selectedWords.forEach((word, wordIndex) => {
            const wordData = allWordsData[word];
            if (!wordData?.temporal_vad?.x || !wordData?.temporal_vad?.v || !wordData?.temporal_vad?.a || !wordData?.temporal_vad?.d) {
                // console.warn(`Missing VAD data for word: ${word}`);
                return;
            }

            const { x, v, a, d } = wordData.temporal_vad;

            if (!timeRef) {
                timeRef = x;
            } else if (JSON.stringify(timeRef) !== JSON.stringify(x)) {
                // console.warn(`Time data mismatch for word: ${word}. Skipping.`);
                return;
            }

            validDataFound = true;

            const colorBaseIndex = wordIndex % lineColors.length;
            const vColor = lineColors[(colorBaseIndex * 3) % lineColors.length];
            const aColor = lineColors[(colorBaseIndex * 3 + 1) % lineColors.length];
            const dColor = lineColors[(colorBaseIndex * 3 + 2) % lineColors.length];

            lines.push({ word: word, dim: 'Valence', color: vColor });
            lines.push({ word: word, dim: 'Arousal', color: aColor });
            lines.push({ word: word, dim: 'Dominance', color: dColor });

            x.forEach((time, index) => {
                if (!combinedChartData[time]) {
                    combinedChartData[time] = { time };
                }
                const vVal = typeof v[index] === 'number' && !isNaN(v[index]) ? v[index] : null;
                const aVal = typeof a[index] === 'number' && !isNaN(a[index]) ? a[index] : null;
                const dVal = typeof d[index] === 'number' && !isNaN(d[index]) ? d[index] : null;

                combinedChartData[time][`${word}_V`] = vVal;
                combinedChartData[time][`${word}_A`] = aVal;
                combinedChartData[time][`${word}_D`] = dVal;

                if (vVal !== null) allValues.push(vVal);
                if (aVal !== null) allValues.push(aVal);
                if (dVal !== null) allValues.push(dVal);

                if (showProp && propData && word === singleWord) {
                    combinedChartData[time]['proportion'] = propData[index];
                }
            });
        });

        const finalChartData = Object.values(combinedChartData).sort((p1, p2) => p1.time - p2.time);
        let finalYDomain: [number | string, number | string] = [0, 1];

        if (allValues.length > 0) {
            const minValue = Math.min(...allValues);
            const maxValue = Math.max(...allValues);
            const range = maxValue - minValue;
            const padding = range < 0.01 ? 0.05 : range * 0.05;
            const calculatedMin = minValue - padding;
            const calculatedMax = maxValue + padding;

            if (calculatedMin < calculatedMax) {
                if (calculatedMin < 0 || calculatedMax > 1) {
                    finalYDomain = [calculatedMin, calculatedMax];
                } else {
                    finalYDomain = [Math.max(0, calculatedMin), Math.min(1, calculatedMax)];
                }
            } else if (allValues.length === 1) {
                finalYDomain = [allValues[0] - padding, allValues[0] + padding];
                finalYDomain = [Math.max(0, finalYDomain[0]), Math.min(1, finalYDomain[1])];
                if(finalYDomain[0] >= finalYDomain[1]) {
                    finalYDomain = [finalYDomain[0] - 0.05, finalYDomain[1] + 0.05];
                }
            }
            if (finalYDomain[0] >= finalYDomain[1]) {
                finalYDomain = [0, 1];
            }
        }

        return { chartData: finalChartData, yDomain: finalYDomain, linesToPlot: lines, hasValidData: validDataFound, timeDataRef: timeRef, showProportionLine: showProp };

    }, [selectedWords, allWordsData, selectedSenseData]);


    if (selectedWords.length === 0) {
        return (
            <div className="plot-placeholder info">
                <h4>Select Word(s)</h4>
                <p>Select one or more words from the controls to view the VAD plot.</p>
            </div>
        );
    }

    if (!hasValidData) {
        return (
            <div className="plot-placeholder error">
                <Icon path={mdiAlertCircleOutline} size={1.6} className="placeholder-icon" />
                <h4>Data Error</h4>
                <p>No valid VAD data found for the selected word(s).</p>
                <p className="subtle">Ensure time data aligns across selected words.</p>
            </div>
        );
    }

    const MultiVADTooltip: React.FC<CustomTooltipProps> = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="vad-custom-tooltip multi-line">
                    <p className="label">{`Year: ${label}`}</p>
                    {payload.map((entry: PayloadEntry, index: number) => {
                        if (entry.dataKey === 'proportion') {
                            const value = entry.value;
                            return (
                                <p key={`${index}-prop`} className="desc proportion" style={{ color: 'var(--bs-secondary-color, #6c757d)' }}>
                                    {`Sense Prop.: ${value?.toFixed(3) ?? 'N/A'}`}
                                </p>
                            );
                        } else {
                            const keyParts = entry.dataKey.split('_');
                            if (keyParts.length < 2) return null;

                            const word = keyParts[0];
                            const dimLetter = keyParts[1];
                            const dimension = dimLetter === 'V' ? 'Valence' : dimLetter === 'A' ? 'Arousal' : 'Dominance';
                            const value = entry.value;
                            const description = getVADDescription(dimension as VADDimension, value);
                            return (
                                <p key={index} className="desc" style={{ color: entry.color }}>
                                    {`${word} ${dimension}: ${value?.toFixed(3) ?? 'N/A'}`}
                                    {description && <span className="qualitative">({description})</span>}
                                </p>
                            );
                        }
                    })}
                </div>
            );
        }
        return null;
    };


    return (
        <div className="plot2d-vad-wrapper">
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
                        label={{ value: 'VAD Value', angle: -90, position: 'insideLeft', dx: -5 }}
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
                        content={<MultiVADTooltip />}
                    />
                    <Legend verticalAlign="top" height={36} />
                    {linesToPlot.map(lineInfo => (
                        <Line
                            yAxisId="left"
                            key={`${lineInfo.word}_${lineInfo.dim}`}
                            type="monotone"
                            dataKey={`${lineInfo.word}_${lineInfo.dim.charAt(0)}`}
                            name={`${lineInfo.word} ${lineInfo.dim.charAt(0)}`}
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

export default Plot2DVAD;