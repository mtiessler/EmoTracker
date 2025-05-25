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
import { CombinedDataPoint, LoadedData, SenseInfo, PredictedVadPoint } from '../types';
import { getVADDescription, VADDimension } from '../vadUtils';

import '../styles/Plot2D.scss';

interface Plot2DProps {
    selectedWords: string[];
    allWordsData: LoadedData | null;
    yLabel: VADDimension;
    selectedSenseData?: SenseInfo | null;
    predictedVadSeries?: PredictedVadPoint[] | null;
}

interface PayloadEntry {
    color: string;
    dataKey: string;
    name: string;
    value: number | null;
    payload: CombinedDataPoint;
    stroke: string;
    fill: string;
    strokeDasharray?: string | number;
}

interface CustomTooltipProps extends TooltipProps<number | null, string> {
    active?: boolean;
    payload?: PayloadEntry[];
    label?: string | number;
    dimension?: VADDimension;
}

interface ExtendedLineInfo2D {
    word: string;
    dataKey: string;
    color: string;
    name: string;
    isForecast?: boolean;
}

const lineColors = ['#0d6efd', '#6f42c1', '#198754', '#ffc107', '#dc3545', '#fd7e14', '#20c997', '#6610f2'];
const forecastLineStylePlot2D = { strokeDasharray: "8 4", strokeWidth: 2 };


const Plot2D: React.FC<Plot2DProps> = ({ selectedWords, allWordsData, yLabel, selectedSenseData, predictedVadSeries }) => {

    const { chartData, yDomain, linesToPlot, hasValidData, timeDataRef, showProportionLine } = useMemo(() => {
        const workingChartDataMap = new Map<number, CombinedDataPoint>();
        const allDimValues: number[] = [];
        const lines: ExtendedLineInfo2D[] = [];
        let commonTimeRef: number[] | null = null;
        let validDataFound = false;
        const dimKey = yLabel.charAt(0).toLowerCase() as 'v' | 'a' | 'd';
        let propData: (number | null)[] | null = null;
        let showProp = false;

        if (!allWordsData || selectedWords.length === 0 || !['v', 'a', 'd'].includes(dimKey)) {
            return { chartData: [], yDomain: [0, 1] as [number | string, number | string], linesToPlot: [], hasValidData: false, timeDataRef: ['auto'], showProportionLine: false };
        }

        const singleWord = selectedWords.length === 1 ? selectedWords[0] : null;
        if (singleWord && selectedSenseData?.y_fitting) {
            const wordDataCheck = allWordsData[singleWord];
            if (wordDataCheck?.temporal_vad?.x && selectedSenseData.y_fitting.length === wordDataCheck.temporal_vad.x.length) {
                propData = selectedSenseData.y_fitting.map(p => (typeof p === 'number' && !isNaN(p) ? p : null));
                if (propData.some(p => p !== null)) {
                    showProp = true;
                }
            }
        }

        selectedWords.forEach((word, wordIndex) => {
            const wordData = allWordsData[word];
            const vadData = wordData?.temporal_vad;

            if (!vadData?.x || !vadData[dimKey] || vadData[dimKey] === undefined) {
                return;
            }

            const x = vadData.x;
            const valueData = vadData[dimKey] as (number | null)[];


            if (!commonTimeRef) {
                commonTimeRef = [...x];
            } else {
                const currentWordTimesMatch = x.length === commonTimeRef.length && x.every((val, idx) => val === (commonTimeRef as number[])[idx]);
                if (!currentWordTimesMatch && !predictedVadSeries) {
                    console.warn(`Time data mismatch for word: ${word} in Plot2D. Skipping.`);
                    return;
                }
            }


            validDataFound = true;
            const color = lineColors[wordIndex % lineColors.length];
            const dataKey = `${word}_${dimKey}`;
            lines.push({ word: word, dataKey: dataKey, color: color, name: word, isForecast: false });

            x.forEach((time, index) => {
                let point = workingChartDataMap.get(time) || { time };
                const dimVal = typeof valueData[index] === 'number' && !isNaN(valueData[index] as number) ? valueData[index] as number : null;
                point[dataKey] = dimVal;
                if (dimVal !== null) allDimValues.push(dimVal);

                if (showProp && propData && word === singleWord) {
                    point['proportion'] = propData[index];
                }
                workingChartDataMap.set(time, point);
            });
        });

        let finalChartData = Array.from(workingChartDataMap.values()).sort((p1,p2) => p1.time - p2.time);

        if (predictedVadSeries && singleWord && commonTimeRef) {
            const word = singleWord;
            const lastActualTimeInData = Math.max(...commonTimeRef.filter(t => typeof t === 'number'));
            const actualWordData = allWordsData?.[word]?.temporal_vad;
            let bridgePointCreated = false;

            if (actualWordData) {
                const dimKeyActual = yLabel.charAt(0).toLowerCase() as 'v' | 'a' | 'd';
                const actualValueArray = actualWordData[dimKeyActual] as (number | null)[];

                predictedVadSeries.forEach((pred) => {
                    let point = workingChartDataMap.get(pred.time) || { time: pred.time };
                    const predValue = pred[dimKey];

                    if (!bridgePointCreated && pred.time > lastActualTimeInData) {
                        const lastActualDataIndex = actualWordData.x.indexOf(lastActualTimeInData);
                        if (lastActualDataIndex !== -1 && actualValueArray) {
                            let bridgeDataPoint = workingChartDataMap.get(lastActualTimeInData) || { time: lastActualTimeInData };
                            bridgeDataPoint[`${word}_${dimKey}_forecast`] = actualValueArray[lastActualDataIndex];
                            workingChartDataMap.set(lastActualTimeInData, bridgeDataPoint);
                        }
                        bridgePointCreated = true;
                    }

                    point[`${word}_${dimKey}_forecast`] = predValue;
                    workingChartDataMap.set(pred.time, point);

                    if (typeof predValue === 'number' && !isNaN(predValue)) allDimValues.push(predValue);
                    if (!commonTimeRef?.includes(pred.time)) {
                        commonTimeRef?.push(pred.time);
                    }
                });
            }

            commonTimeRef?.sort((a,b) => a-b);

            const forecastColor = lines.find(l => l.word === word)?.color || lineColors[selectedWords.indexOf(word) % lineColors.length];
            lines.push({
                word: word,
                dataKey: `${word}_${dimKey}_forecast`,
                color: forecastColor,
                name: `${word} (Forecast)`,
                isForecast: true
            });

            finalChartData = Array.from(workingChartDataMap.values()).sort((p1,p2) => p1.time - p2.time);
        }


        let finalYDomain: [number | string, number | string] = [0, 1];
        if (allDimValues.length > 0) {
            const minValue = Math.min(...allDimValues);
            const maxValue = Math.max(...allDimValues);
            const range = maxValue - minValue;
            const padding = range < 0.01 ? 0.05 : range * 0.05;
            let calculatedMin = minValue - padding;
            let calculatedMax = maxValue + padding;

            if (calculatedMax - calculatedMin < 0.1) {
                const mid = (calculatedMin + calculatedMax) / 2;
                calculatedMin = mid - 0.05;
                calculatedMax = mid + 0.05;
            }

            if (yLabel === 'Valence' || yLabel === 'Arousal' || yLabel === 'Dominance') {
                finalYDomain = [
                    Math.min(calculatedMin, 0),
                    Math.max(calculatedMax, 1)
                ];
            } else {
                finalYDomain = [calculatedMin, calculatedMax];
            }

            if (finalYDomain[0] >= finalYDomain[1]) {
                const center = (finalYDomain[0] + finalYDomain[1]) / 2 || 0.5;
                finalYDomain = [center - 0.1, center + 0.1];
            }
            if (finalYDomain[0] >= finalYDomain[1]) finalYDomain = [0,1];
        }


        return {
            chartData: finalChartData,
            yDomain: finalYDomain,
            linesToPlot: lines,
            hasValidData: validDataFound,
            timeDataRef: commonTimeRef || ['auto'],
            showProportionLine: showProp
        };

    }, [selectedWords, allWordsData, yLabel, selectedSenseData, predictedVadSeries]);


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
                        const isForecast = entry.dataKey?.endsWith('_forecast');
                        const nameSuffix = isForecast ? ' (Forecast)' : '';

                        if (entry.dataKey === 'proportion') {
                            const value = entry.value;
                            content = (
                                <p key={`${index}-prop`} className="desc proportion" style={{ color: 'var(--bs-secondary-color, #6c757d)' }}>
                                    {`Sense Prop.: ${value?.toFixed(3) ?? 'N/A'}`}
                                </p>
                            );
                        } else {
                            const wordFromName = entry.name.replace(' (Forecast)', '');
                            const value = entry.value;
                            const description = getVADDescription(dimension, value);
                            content = (
                                <p key={index} className="desc" style={{ color: entry.color, fontStyle: isForecast ? 'italic' : 'normal' }}>
                                    {`${wordFromName}${nameSuffix}: ${value?.toFixed(3) ?? 'N/A'}`}
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
                        label={{ value: "Year", position: "insideBottom", dy: 15 }}
                        height={50}
                        scale="time"
                        interval="preserveStartEnd"
                        tickLine={false}
                        ticks={Array.isArray(timeDataRef) && timeDataRef.length > 1 && typeof timeDataRef[0] === 'number' ? timeDataRef as unknown as number[]: undefined}
                    />
                    <YAxis
                        yAxisId="left"
                        domain={yDomain}
                        label={{ value: yLabel, angle: -90, position: 'insideLeft', dx: -5 }}
                        width={65}
                        tickFormatter={(tick) => typeof tick === 'number' ? tick.toFixed(2) : String(tick)}
                        allowDataOverflow={true}
                        tickLine={false}
                    />
                    {showProportionLine && (
                        <YAxis
                            yAxisId="right"
                            orientation="right"
                            domain={[0, 1]}
                            label={{ value: 'Sense Prop.', angle: 90, position: 'insideRight', dx: 5 }}
                            width={65}
                            tickFormatter={(tick) => typeof tick === 'number' ? tick.toFixed(2) : String(tick)}
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
                            name={lineInfo.name}
                            stroke={lineInfo.color}
                            strokeWidth={lineInfo.isForecast ? forecastLineStylePlot2D.strokeWidth : 1.5}
                            strokeDasharray={lineInfo.isForecast ? forecastLineStylePlot2D.strokeDasharray : undefined}
                            activeDot={{ r: 4, strokeWidth: 0, fill: lineInfo.color }}
                            dot={lineInfo.isForecast ? { r: 3, fill:lineInfo.color, strokeWidth:0 } : false}
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