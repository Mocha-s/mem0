"use client";

import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { SaveIcon, RotateCcw } from "lucide-react"
import { FormView } from "@/components/form-view"
import { JsonEditor } from "@/components/json-editor"
import { FeatureManager } from "@/components/mem0/FeatureManager"
import { FeedbackManager } from "@/components/mem0/FeedbackWidget"
import { ServiceMonitor } from "@/components/system/ServiceMonitor"
import { SyncManager } from "@/components/sync/SyncManager"
import ConfigGenerator from "@/components/mcp/ConfigGenerator"
import { useConfig } from "@/hooks/useConfig"
import { useSelector } from "react-redux"
import { RootState } from "@/store/store"
import { useToast } from "@/components/ui/use-toast"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"

export default function SettingsPage() {
  const { toast } = useToast()
  const configState = useSelector((state: RootState) => state.config)
  const [settings, setSettings] = useState({
    openmemory: configState.openmemory || {
      custom_instructions: null
    },
    mem0: configState.mem0
  })
  const [viewMode, setViewMode] = useState<"form" | "json">("form")
  const [activeTab, setActiveTab] = useState("config")
  const { fetchConfig, saveConfig, resetConfig, isLoading, error } = useConfig()

  useEffect(() => {
    // Load config from API on component mount
    const loadConfig = async () => {
      try {
        await fetchConfig()
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to load configuration",
          variant: "destructive",
        })
      }
    }
    
    loadConfig()
  }, [])

  // Update local state when redux state changes
  useEffect(() => {
    setSettings(prev => ({
      ...prev,
      openmemory: configState.openmemory || { custom_instructions: null },
      mem0: configState.mem0
    }))
  }, [configState.openmemory, configState.mem0])

  const handleSave = async () => {
    try {
      await saveConfig({ 
        openmemory: settings.openmemory,
        mem0: settings.mem0 
      })
      toast({
        title: "Settings saved",
        description: "Your configuration has been updated successfully.",
      })
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save configuration",
        variant: "destructive",
      })
    }
  }

  const handleReset = async () => {
    try {
      await resetConfig()
      toast({
        title: "Settings reset",
        description: "Configuration has been reset to default values.",
      })
      await fetchConfig()
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to reset configuration",
        variant: "destructive",
      })
    }
  }

  return (
    <div className="text-white py-6">
      <div className="container mx-auto py-10 max-w-6xl">
        <div className="flex justify-between items-center mb-8">
          <div className="animate-fade-slide-down">
            <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
            <p className="text-muted-foreground mt-1">管理OpenMemory和Mem0的配置、功能特性和系统监控</p>
          </div>
          {activeTab === "config" && (
            <div className="flex space-x-2">
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="outline" className="border-zinc-800 text-zinc-200 hover:bg-zinc-700 hover:text-zinc-50 animate-fade-slide-down" disabled={isLoading}>
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Reset Defaults
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Reset Configuration?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will reset all settings to the system defaults. Any custom configuration will be lost.
                      API keys will be set to use environment variables.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleReset} className="bg-red-600 hover:bg-red-700">
                      Reset
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
              
              <Button onClick={handleSave} className="bg-primary hover:bg-primary/90 animate-fade-slide-down" disabled={isLoading}>
                <SaveIcon className="mr-2 h-4 w-4" />
                {isLoading ? "Saving..." : "Save Configuration"}
              </Button>
            </div>
          )}
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full animate-fade-slide-down delay-1">
          <TabsList className="grid w-full grid-cols-6 mb-8">
            <TabsTrigger value="config">基础配置</TabsTrigger>
            <TabsTrigger value="features">功能特性</TabsTrigger>
            <TabsTrigger value="feedback">反馈管理</TabsTrigger>
            <TabsTrigger value="monitor">系统监控</TabsTrigger>
            <TabsTrigger value="sync">数据同步</TabsTrigger>
            <TabsTrigger value="advanced">高级设置</TabsTrigger>
          </TabsList>

          <TabsContent value="config">
            <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as "form" | "json")} className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-8">
                <TabsTrigger value="form">Form View</TabsTrigger>
                <TabsTrigger value="json">JSON Editor</TabsTrigger>
              </TabsList>

              <TabsContent value="form">
                <FormView settings={settings} onChange={setSettings} />
              </TabsContent>

              <TabsContent value="json">
                <Card>
                  <CardHeader>
                    <CardTitle>JSON Configuration</CardTitle>
                    <CardDescription>Edit the entire configuration directly as JSON</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <JsonEditor value={settings} onChange={setSettings} />
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </TabsContent>

          <TabsContent value="features">
            <FeatureManager />
          </TabsContent>

          <TabsContent value="feedback">
            <FeedbackManager />
          </TabsContent>

          <TabsContent value="monitor">
            <ServiceMonitor />
          </TabsContent>

          <TabsContent value="sync">
            <SyncManager />
          </TabsContent>

          <TabsContent value="advanced">
            <div className="space-y-6">
              {/* MCP客户端配置 */}
              <ConfigGenerator />
              
              <Card>
                <CardHeader>
                  <CardTitle>高级配置选项</CardTitle>
                  <CardDescription>
                    高级用户配置选项和系统调优参数
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <Card className="border-zinc-800">
                        <CardHeader className="pb-3">
                          <CardTitle className="text-base">API配置</CardTitle>
                          <CardDescription className="text-sm">
                            API端点和超时设置
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div>
                            <label className="text-sm font-medium">OpenMemory API URL</label>
                            <p className="text-sm text-muted-foreground">
                              {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765"}
                            </p>
                          </div>
                          <div>
                            <label className="text-sm font-medium">Mem0 API URL</label>
                            <p className="text-sm text-muted-foreground">
                              {process.env.NEXT_PUBLIC_MEM0_API_URL || "http://localhost:8000"}
                            </p>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="border-zinc-800">
                        <CardHeader className="pb-3">
                          <CardTitle className="text-base">缓存设置</CardTitle>
                          <CardDescription className="text-sm">
                            缓存策略和过期时间
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div>
                            <label className="text-sm font-medium">配置缓存时间</label>
                            <p className="text-sm text-muted-foreground">5分钟</p>
                          </div>
                          <div>
                            <label className="text-sm font-medium">搜索结果缓存</label>
                            <p className="text-sm text-muted-foreground">启用</p>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="border-zinc-800">
                        <CardHeader className="pb-3">
                          <CardTitle className="text-base">安全设置</CardTitle>
                          <CardDescription className="text-sm">
                            安全策略和访问控制
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div>
                            <label className="text-sm font-medium">API密钥验证</label>
                            <p className="text-sm text-muted-foreground">启用</p>
                          </div>
                          <div>
                            <label className="text-sm font-medium">CORS设置</label>
                            <p className="text-sm text-muted-foreground">受限制</p>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="border-zinc-800">
                        <CardHeader className="pb-3">
                          <CardTitle className="text-base">性能优化</CardTitle>
                          <CardDescription className="text-sm">
                            性能调优和资源限制
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div>
                            <label className="text-sm font-medium">并发连接数</label>
                            <p className="text-sm text-muted-foreground">最大100</p>
                          </div>
                          <div>
                            <label className="text-sm font-medium">请求超时</label>
                            <p className="text-sm text-muted-foreground">30秒</p>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    <Card className="border-zinc-800">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-base">环境变量</CardTitle>
                        <CardDescription className="text-sm">
                          当前环境变量配置（只读）
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid gap-2 text-sm font-mono">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">NEXT_PUBLIC_API_URL:</span>
                            <span>{process.env.NEXT_PUBLIC_API_URL || '未设置'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">NEXT_PUBLIC_MEM0_API_URL:</span>
                            <span>{process.env.NEXT_PUBLIC_MEM0_API_URL || '未设置'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">NODE_ENV:</span>
                            <span>{process.env.NODE_ENV || '未设置'}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
